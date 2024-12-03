import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time

# Load YOLO model
model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Define the video path
video_path = r'D:\computer vision project\Car-speed-estimation-and-counting-using-YOLOv8-computer-vision-main\veh2.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read class names from coco.txt
with open(r"D:\computer vision project\Car-speed-estimation-and-counting-using-YOLOv8-computer-vision-main\coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

# Initialize variables
count = 0
tracker = Tracker()

cy1 = 322  # Line 1 (can be adjusted based on your video)
cy2 = 368  # Line 2 (can be adjusted based on your video)
offset = 6

# Dictionaries to track the time when cars cross the lines
vh_right = {}
vh_left = {}

# Lists to store counted cars going in both directions
counter_right = []
counter_left = []

# Assuming the distance between lines is 10 meters
distance = 10  # meters

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame or end of video reached.")
        break
    count += 1
    if count % 3 != 0:  # Skip some frames to reduce processing load
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO inference
    results = model.predict(frame)
    
    # Access bounding boxes and other results from YOLOv8 prediction
    boxes = results[0].boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)
    confs = results[0].boxes.conf  # Confidence scores
    classes = results[0].boxes.cls  # Class labels (as integers)

    list = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        confidence = confs[i].item()
        class_id = int(classes[i].item())
        c = class_list[class_id]
        if 'car' in c:  # Only track cars
            list.append([int(x1), int(y1), int(x2), int(y2)])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        
        # Draw a red bounding box around the car
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Red color box (BGR format)

        # Track cars going to the right (left to right)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_right[id] = time.time()

        if id in vh_right:
            if cy2 < (cy + offset) and cy2 > (cy - offset):  # Crossing the second line (right direction)
                elapsed_time = time.time() - vh_right[id]
                if id not in counter_right:
                    counter_right.append(id)
                    speed = distance / elapsed_time * 3.6  # Convert speed to km/h
                    cv2.putText(frame, f"ID:{id} Speed: {int(speed)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Track cars going to the left (right to left)
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_left[id] = time.time()

        if id in vh_left:
            if cy1 < (cy + offset) and cy1 > (cy - offset):  # Crossing the first line (left direction)
                elapsed_time = time.time() - vh_left[id]
                if id not in counter_left:
                    counter_left.append(id)
                    speed = distance / elapsed_time * 3.6  # Convert speed to km/h
                    cv2.putText(frame, f"ID:{id} Speed: {int(speed)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Draw the lines for tracking
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)

    # Display counts for cars in both directions
    cv2.putText(frame, f"Going Right: {len(counter_right)}", (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Going Left: {len(counter_left)}", (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit when 'Esc' is pressed
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()