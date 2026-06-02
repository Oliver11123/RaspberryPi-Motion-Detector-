# RaspberryPi-Ai-Motion-Detector- 
Hello! (This document is meant to be viewed in print layout)

This is my project about making an AI motion detector with nothing more than a raspberry pi 4b (any raspberry pi works) and a USB camera (a RaspberryPi camera works too). I will try to make this as detailed as possible so ANYONE can re-make this project!

Requirements 
You will need a;
RaspberryPi 
Power supply 
Webcamera or RaspberryPi camera 
MicroSD card
Computer that can read a micro SD card of SD card
Micro SD to SD card adapter 

Introduction 
Beginning with the basics first you need a micro SD Card (16gb recommended) in order to have your Raspberry Pi work. Then you need a micro SD to a SD card adapter(depends on your own case). 

For me in my computer I have a SD card insert so I'll be using a micro SD card to a SD card adapter. Then you want to go to the RaspberryPi official website on your computer and download the RaspberryPi Imager (any will do). Once that's completed you need to go into “Choose OS” then “Other specific-purpose OS” then “3D Printing” then scroll down and find OctoPi and download the stable option. 

Then choose your storage and the device. Ensure it's your RPI* storage as this deletes all past data on the SD card. Once you press the next press edit settings, here you will need to enter your RaspberryPi Username and password. I recommend enabling SSH as this allows you to connect to your pi from your computer without using a cable. Tick the configure wireless LAN box. Change your wireless LAN country to your corresponding country and I recommend that you take a photo or screenshot of this as you will need it for logging into your pi over SSH. 

Once this is done click done and then download the OS on your SD card. Once it's completed downloading then you can put the SD card into the RPI and boot it up by plugging it into electricity.

Starting To Code
This is where we will start to code everything like the motion and the AI

Firstly, we will assign the RPI a specific IP address so that you never have to access another website again on your phone, you just can have it saved as your favourite. If we don't do this step you won't know what the IP is and that means you aren't able to access this website. The website URL is like the following example https//:192.168.1.100

You want to open the terminal and write the following code, Step 1; 


FOR “wlan0)
#Open your network using nmtui 
sudo nmtui

#Press enter on “ Edit a connection”
#I will scroll to wireless WiFi but will do Ethernet after this. 


#You will want to go down to the “IPv4 CONFIGURATION” 
#Then change your “Addresses” to whatever you wish. I will use 192.168.1.100/24

(please change this with numbers you will remember or number a that suits you)

#Change your Gateway to 192.168.1.1
#Change your  DNS Servers to 192.168.1.100 and then under that 8.8.8.8
#Then enter “OK”
#Then enter “Back”
#Then enter “Quit”

#Now just Reboot

sudo reboot



FOR (eth0)

sudo nmtui

#Press enter on “ Edit a connection”
#Scroll to  Ethernet


#Then show your “IPv4 CONFIGURATION”
#Then change your “Addresses” to whatever you wish. I will use 192.168.1.100/24 


#Change your Gateway to 192.168.1.1
#Change your  DNS Servers to 192.168.1.100 and then under that 8.8.8.8
#Then enter “OK”
#Then enter “Back”
#Then enter “Quit”

Note: There might have been an easier way to do this but I couldn't find any code online that could just be copied and paste I need to look into this to make it easier 

Step 2, Now we will Install The Dependencies;

#Updates your RPI (shouldn't be needed but just in case)
sudo apt update && sudo apt upgrade -y

#Install the required libraries for this to work 
sudo apt install -y python3-opencv python3-numpy python3-pip
sudo apt install -y libatlas-base-dev libhdf5-dev

#Install Numpy packages for the libraries
pip3 install opencv-python numpy

This was also an interesting experience as I wanted to install Numpy and a specific version of it but it kept breaking each time so I have to go over this as later I did delete Numpy and it did work? Idk I'll get back to this later

Ps: This may take a few minutes 

This installs all the dependencies for the project. Dependencies are basically the ingredients we need for the recipe, the code is the recipe, and the RPI is the kitchen. That's my analogy at least.

Step 3;

#Makes all the folders for the plugins
mkdir -p ~/octoprint/plugins/ai_detector
mkdir -p ~/octoprint/plugins/ai_detector/templates
mkdir -p ~/octoprint/plugins/ai_detector/static/js
mkdir -p ~/octoprint/plugins/ai_detector/static/css
mkdir -p ~/octoprint/plugins/ai_detector/models



Step 4 will be where we download the AI models from the Internet.

# Opens the file to download the AI’s to
cd ~/octoprint/plugins/ai_detector/models

#Working URL for yolov3-tiny weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

#Or try this mirror if the above fails
wget https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3-tiny.weights?raw=true -O yolov3-tiny.weights

# The ?raw=true is IMPORTANT - without it you download an HTML page, not the weights file!

#Then this
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg 

#Finally this
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
 

#This verifies the downloads
ls -lh ~/octoprint/plugins/ai_detector/models/
# You should see: yolov3-tiny.cfg, coco.names and yolov3-tiny.weights

Note: this step was not great I had to try multiple ways to download different AI's and everyone who had previously made good AI’s that identify things changed the file names and directories and I couldn't didn't working ones for a minute 

Step 5 then we have to create a python file. In the terminal write this

nano ai_motion_detector.py

Step 6 we copy and paste the following long code into the file

```

#!/usr/bin/env python3
import cv2
import numpy as np
from datetime import datetime
import time
import os
import signal
import sys

class AIMotionDetector:
    def __init__(self):
        print("🚀 Initializing AI Motion Detector...")
        
        # Get the directory where THIS script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths to your downloaded models (now works automatically!)
        self.weights_path = os.path.join(script_dir, "yolov3-tiny.weights")
        self.cfg_path = os.path.join(script_dir, "yolov3-tiny.cfg")
        self.names_path = os.path.join(script_dir, "coco.names")
        
        # Check if files exist
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
        
        # Load YOLO
        print("📦 Loading YOLO model...")
        self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
        
        # Use optimized backend for Raspberry Pi
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load class names
        with open(self.names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        print(f"✓ Loaded {len(self.classes)} object classes")
        
        # Motion detection setup
        print("🎥 Initializing camera...")
        self.cap = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream')
        if not self.cap.isOpened():
            print("❌ ERROR: Could not open camera")
            sys.exit(1)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera opened successfully")
        
        # Background subtractor for motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
        
        # Settings
        self.motion_threshold = 3000  # Lower = more sensitive
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.frame_skip = 2  # Process every 2nd frame for performance
        self.frame_count = 0
        
        # Statistics
        self.total_detections = 0
        self.motion_events = 0
        
        print("✅ Ready! Press 'q' to quit, 's' for snapshot, '+'/'-' for sensitivity")
        
    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        fgmask = self.fgbg.apply(frame)
        
        # Remove shadows
        fgmask = np.where(fgmask == 127, 0, fgmask).astype(np.uint8)
        
        # Noise reduction
        fgmask = cv2.medianBlur(fgmask, 5)
        
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Ignore tiny movements (noise)
                motion_area += area
                # Draw motion contours in debug mode
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)
        
        return motion_area > self.motion_threshold, motion_area
    
    def detect_objects(self, frame):
        """Detect specific objects using YOLO"""
        height, width = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get detections
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
        
        # Apply Non-Maximum Suppression
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
        # Status bar
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), status_color, -1)
        
        # Status text
        status = "🔴 MOTION DETECTED" if motion_detected else "🟢 Monitoring"
        cv2.putText(frame, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Stats
        stats = f"Motion Area: {motion_area} | Detections: {self.total_detections}"
        cv2.putText(frame, stats, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw object boxes
        for obj in objects:
            x, y, w, h = obj['bbox']
            # Color based on object type
            if obj['class'] == 'person':
                color = (0, 0, 255)  # Red for people
            elif obj['class'] in ['car', 'truck', 'bus', 'motorbike', 'bicycle']:
                color = (255, 165, 0)  # Orange for vehicles
            else:
                color = (0, 255, 0)  # Green for other objects
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Label with background
            label = f"{obj['class']}: {obj['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 5), (x + label_size[0] + 5, y), color, -1)
            cv2.putText(frame, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (frame.shape[1] - 100, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def save_snapshot(self, frame, motion_detected, objects):
        """Save a snapshot with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        
        # Add text overlay
        snapshot = frame.copy()
        cv2.putText(snapshot, timestamp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if objects:
            objects_text = ", ".join([f"{obj['class']}" for obj in objects[:5]])
            cv2.putText(snapshot, f"Objects: {objects_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(filename, snapshot)
        print(f"📸 Snapshot saved: {filename}")
        return filename
    
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
        snapshot_interval = 10  # Seconds between auto-snapshots during motion
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % self.frame_skip != 0:
                    # Still show frame but skip processing
                    cv2.imshow('AI Motion Detector', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect motion
                motion_detected, motion_area = self.detect_motion(gray)
                
                objects = []
                if motion_detected:
                    self.motion_events += 1
                    # Run object detection only when motion is detected
                    objects = self.detect_objects(frame)
                    self.total_detections += len(objects)
                    
                    # Auto-save snapshot during motion (cooldown)
                    current_time = time.time()
                    if current_time - last_snapshot_time > snapshot_interval:
                        self.save_snapshot(frame, motion_detected, objects)
                        last_snapshot_time = current_time
                    
                    # Print to console
                    if objects:
                        print(f"🚨 MOTION! Found: {', '.join([o['class'] for o in objects])}")
                
                # Draw everything on frame
                frame = self.draw_detections(frame, objects, motion_detected, motion_area)
                
                # Display
                cv2.imshow('AI Motion Detector', frame)
                
                # Handle keyboard input
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
```

Note: This AI script is half things I did in school and half edited half copied from other people and it's very complicated I don't think I wanna get into explaining everything but I might 😭 FAHH

Then press ctrl x, y and then enter.

Then in the terminal use this code and ensure you're still in the correct folder

python3 ai_motion.py

Photo of what I see:



Ensure your camera is plugged in and it should work! It opens a separate terminal with what the camera sees and the following image under is what shows on the other terminal. You can access the photos using the following command

cd /home/raspberrypi/OctopPrint/plugins/ai_detector/models

Note: now that I rebooted it it seems to not work. I think it is because both OctopPrint and the AI motion detector are trying to access the same camera at the same time. This is an underlying factor however I believe it could be resolved through mirroring what one camera sees to a virtual camera view. Idk its act gonna work tho 💀
Note 2: NVM I fixed it it was an issue with the two scripts accessing the same camera at the same time, the solution was to tell the ai script to access the image from the OctopPrint page.




If you want to stop the process do ctrl c


If you reboot use this code to have easier access to go into and run the code

cd ~/octoprint/plugins/ai_detector/models
python3 ai_motion.py

Troubleshooting 
Camera not found -> check your USB physical connection
Device or resource busy -> use HTTP stream link instead from OctopPrint 
Can't find YOLO files -> download the wget files again

The project is done now but I documented the entirety of the things I did and I did run into issues so don't copy anything after here.

New things I'm trying for the solution of 2 scripts trying to use one camera

#installs something to mirror the camera
sudo apt update
sudo apt install v4l2loopback-dkms
Didn't work
#this makes a virtual path of video 10 mirror image
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1
Didn't work 
ffmpeg -f v4l2 -i /dev/video0 -f v4l2 /dev/video17
Worked but didn't resolve just confirmed both scripts were using one camera port
cap = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream')
This ended up working. It makes the AI script use the OctopPrint HTML site to use it for AI motion detection 

Extra:
if you want to add a video thing add this;


Sources;
RPI* = RaspberryPi
RPI imager; Raspberry Pi software – Raspberry Pi https://share.google/xWMGkLNPoBPmaxi7i 
