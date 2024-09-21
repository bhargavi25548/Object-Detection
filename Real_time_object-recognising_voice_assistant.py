import cv2
import numpy as np
import os
import time
from gtts import gTTS
from playsound import playsound

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Loading coco.names file, which contains names of objects it can detect
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open webcam
cap = cv2.VideoCapture(0)  # You can change 0 to 1 or 2 depending on your webcam index
frame_no = 0
inc = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    height, width, channels = frame.shape  # Get frame dimensions

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to avoid overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.6, nms_threshold=0.5)

    # Draw bounding boxes and display object labels
    object_detected = False
    for i in range(len(boxes)):
        if i in indexes:
            object_detected = True
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Print label in the output shell
            print(f"Detected: {label} with confidence {confidence:.2f}")

            # Draw bounding box and label on the frame
            color = (0, 255, 0)  # BGR format (green color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert label to voice
            voice_text = f"{label} in front of you"
            file_path = f'voice_{inc}.mp3'

            inc += 1

            # Generate voice output
            tts = gTTS(text=voice_text, lang='en')
            tts.save(file_path)

            # Play voice output with error handling
            try:
                playsound(file_path)
            except Exception as e:
                print(f"Error playing sound: {e}")

            # Clean up - remove temporary voice file
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file: {e}")

    # If no object detected, print a message
    if not object_detected:
        print("No object detected")

    # Calculate FPS (Frames Per Second)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Exit key (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
