#!/usr/bin/env python3
#Imports all of the necessary libraries that you need for this project
import cv2
import numpy as np
from datetime import datetime
import time
import os
import signal
import sys

#Assigns a specific word a class
class AIMotionDetector:
    def __init__(self):
        print("🚀 Initializing AI Motion Detector...")
        
        #Get the directory where THIS script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        #Paths to your downloaded models 
        self.weights_path = os.path.join(script_dir, "yolov3-tiny.weights")
        self.cfg_path = os.path.join(script_dir, "yolov3-tiny.cfg")
        self.names_path = os.path.join(script_dir, "coco.names")
        
        #Check if files exist
        if not os.path.exists(self.weights_path):
            print(f"❌ ERROR: weights file not found at {self.weights_path}")
            sys.exit(1)
        if not os.path.exists(self.cfg_path):
            print(f"❌ ERROR: config file not found at {self.cfg_path}")
            sys.exit(1)
        if not os.path.exists(self.names_path):
            print(f"❌ ERROR: names file not found at {self.names_path}")
            sys.exit(1)
        
        print("✓ All model files found")
        
        #Load YOLO (the AI model)
        print("📦 Loading YOLO model...")
        self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
        
        #Use optimized backend for Raspberry Pi
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        #Load class names
        with open(self.names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print(f"✓ Loaded {len(self.classes)} object classes")
        
        #Motion detection setup
        print("🎥 Initializing camera...")
        self.cap = cv2.VideoCapture('http://192.168.0.248/webcam/?action=stream')
        
        if not self.cap.isOpened():
            print("❌ ERROR: Could not open camera")
            sys.exit('http://192.168.0.248/webcam/?action=stream')
        
        #Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera opened successfully")
        
        #Background subtractor for motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
        
        #Settings
        self.motion_threshold = 3000  #Lower = more sensitive
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.frame_skip = 2  #Process every 2nd frame for performance
        self.frame_count = 0
        
        #Statistics
        self.total_detections = 0
        self.motion_events = 0
        
        print("✅ Ready! Press 'q' to quit, 's' for snapshot, '+'/'-' for sensitivity")
        
    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        fgmask = self.fgbg.apply(frame)
        
        #Remove shadows
        fgmask = np.where(fgmask == 127, 0, fgmask).astype(np.uint8)
        
        #Noise reduction
        fgmask = cv2.medianBlur(fgmask, 5)
        
        #Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  #Ignore tiny movements (noise)
                motion_area += area
                #Draw motion contours in debug mode
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)
        
        return motion_area > self.motion_threshold, motion_area
    
    def detect_objects(self, frame):
        """Detect specific objects using YOLO"""
        height, width = frame.shape[:2]
        
        #Create blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        #Get detections
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        detections = self.net.forward(output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        #Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        objects = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                objects.append({
                    'class': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': (x, y, w, h)
                })
        
        return objects
    
    def draw_detections(self, frame, objects, motion_detected, motion_area):
        """Draw all annotations on frame"""
        #Status bar
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), status_color, -1)
        
        #Status text
        status = "🔴 MOTION DETECTED" if motion_detected else "🟢 Monitoring"
        cv2.putText(frame, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        #Stats
        stats = f"Motion Area: {motion_area} | Detections: {self.total_detections}"
        cv2.putText(frame, stats, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        #Draw object boxes
        for obj in objects:
            x, y, w, h = obj['bbox']
            #Color based on object type
            if obj['class'] == 'person':
                color = (0, 0, 255)  #Red for people
            elif obj['class'] in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                color = (255, 165, 0)  #Orange for vehicles
            else:
                color = (0, 255, 0)  #Green for other objects
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            #Label with background
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 5), (x + label_size[0] + 5, y), color, -1)
            cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        #FPS counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (frame.shape[1] - 100, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def save_snapshot(self, frame, motion_detected, objects):
        """Save a snapshot with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        
        #Add text overlay
        snapshot = frame.copy()
        cv2.putText(snapshot, timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if objects:
            objects_text = ", ".join([f"{obj['class']}" for obj in objects[:5]])
            cv2.putText(snapshot, f"Objects: {objects_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(filename, snapshot)
        print(f"📸 Snapshot saved: {filename}")
        return filename
    
    #Runs the main script to show everything
    def run(self):
        """Main loop"""
        print("\n🎯 AI Motion Detector Running")
        print("=" * 40)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save snapshot")
        print("  '+' - Increase sensitivity")
        print("  '-' - Decrease sensitivity")
        print("=" * 40 + "\n")
        
        last_snapshot_time = 0
        snapshot_interval = 10  #Seconds between auto-snapshots during motion
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                #Skip frames for performance
                if self.frame_count % self.frame_skip != 0:
                    #Still show frame but skip processing
                    cv2.imshow('AI Motion Detector', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                #Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                #Detect motion
                motion_detected, motion_area = self.detect_motion(gray)
                
                objects = []
                if motion_detected:
                    self.motion_events += 1
                    #Run object detection only when motion is detected
                    objects = self.detect_objects(frame)
                    self.total_detections += len(objects)
                    
                    #Auto-save snapshot during motion (cooldown)
                    current_time = time.time()
                    if current_time - last_snapshot_time > snapshot_interval:
                        self.save_snapshot(frame, motion_detected, objects)
                        last_snapshot_time = current_time
                    
                    #Print to console
                    if objects:
                        print(f"🚨 MOTION! Found: {', '.join([o['class'] for o in objects])}")
                
                #Draw everything on frame
                frame = self.draw_detections(frame, objects, motion_detected, motion_area)
                
                #Display
                cv2.imshow('AI Motion Detector', frame)
                
                #Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Shutting down...")
                    break
                elif key == ord('s'):
                    self.save_snapshot(frame, motion_detected, objects)
                elif key == ord('+') or key == ord('='):
                    self.motion_threshold = max(500, self.motion_threshold - 500)
                    print(f"⚡ Sensitivity increased (threshold: {self.motion_threshold})")
                elif key == ord('-') or key == ord('_'):
                    self.motion_threshold = min(20000, self.motion_threshold + 500)
                    print(f"🐢 Sensitivity decreased (threshold: {self.motion_threshold})")
                    
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print(f"\n📊 Final Statistics:")
        print(f"   - Motion events: {self.motion_events}")
        print(f"   - Objects detected: {self.total_detections}")
        print(f"   - Total frames: {self.frame_count}")
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    detector = AIMotionDetector()
    detector.run()
