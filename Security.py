# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# import time

# cap = cv2.VideoCapture(0)
# cap.set(3, 1000)
# cap.set(4, 640)

# model = YOLO("yolov8n.pt")

# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]

# # Classes to blur (screen-related classes)
# screen_classes = ["tvmonitor", "laptop", "cell phone"]

# prev_frame_time = 0
# new_frame_time = 0

# while True:
#     new_frame_time = time.time()
#     success, img = cap.read()
#     if not success:
#         break

#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
            
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1

#             cls = int(box.cls[0])
#             class_name = classNames[cls]

#             if class_name in screen_classes:
#                 screen_region = img[y1:y2, x1:x2]
#                 blurred_region = cv2.GaussianBlur(screen_region, (51, 51), 0)
#                 img[y1:y2, x1:x2] = blurred_region

#                 cvzone.cornerRect(img, (x1, y1, w, h))
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

#     fps = 1 / (new_frame_time - prev_frame_time)
#     prev_frame_time = new_frame_time

#     cv2.imshow("Webcam Detection", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1000)  # Set the width
cap.set(4, 640)   # Set the height

# Load the YOLO model
model = YOLO("yolov8n.pt")

# List of class names to detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Classes to blur (screen-related classes)
screen_classes = ["tvmonitor", "laptop", "cell phone"]

prev_frame_time = 0
new_frame_time = 0

security_enabled = False

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in screen_classes:
                # Apply blurring to the detected screen region
                screen_region = img[y1:y2, x1:x2]
                blurred_region = cv2.GaussianBlur(screen_region, (51, 51), 0)
                img[y1:y2, x1:x2] = blurred_region
                security_enabled = True

            # Draw bounding box and label for all detected objects
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Display security status
    if security_enabled:
        cv2.putText(img, "Security Enabled", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(img, "No need of Security", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 100), scale=2, thickness=2)

    # Display the result
    cv2.imshow("Webcam Detection", img)
    
    # Reset security status if no screen-related object is detected
    if not any(box.cls[0] in screen_classes for r in results for box in r.boxes):
        security_enabled = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
