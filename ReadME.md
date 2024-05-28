# CCTV Security with YOLOv8

This project demonstrates a real-time computer vision application using the YOLOv8 model to detect and blur screens in a camera feed. The detected screens (such as TV monitors, laptops, and cell phones) are automatically blurred to protect sensitive information or enhance privacy.

## Features

- Real-time detection of screens using the YOLOv8 model.
- Blurring of detected screen regions.
- Display of bounding boxes and confidence scores for detected screens.

## Installation

### Requirements

- Python 3.6+
- OpenCV
- Ultralytics YOLOv8
- CVZone

### Install Dependencies

```bash
pip install ultralytics opencv-python cvzone
```

## Usage

1. **Clone the Repository** (if applicable):

   ```bash
   git clone https://github.com/musharrafhamraz/CCTV-Security.git
   cd CCTV-Security
   ```

2. **Run the Script**:

   ```bash
   python Security.py
   ```

## Code Explanation

The script captures video from the webcam, detects screens in real-time, and blurs the detected regions.

### Important Sections

1. **Importing Libraries**:

   ```python
   from ultralytics import YOLO
   import cv2
   import cvzone
   import math
   import time
   ```

2. **Initializing Video Capture and YOLO Model**:

   ```python
   cap = cv2.VideoCapture(0)
   cap.set(3, 1000)
   cap.set(4, 640)
   
   model = YOLO("yolov8n.pt")
   ```

3. **Class Names and Screen Classes**:

   ```python
   classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", ...]
   screen_classes = ["tvmonitor", "laptop", "cell phone"]
   ```

4. **Main Loop for Real-Time Detection**:

   ```python
   while True:
       new_frame_time = time.time()
       success, img = cap.read()
       if not success:
           break

       results = model(img, stream=True)
       for r in results:
           boxes = r.boxes
           for box in boxes:
               x1, y1, x2, y2 = box.xyxy[0]
               x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
               w, h = x2 - x1, y2 - y1

               cls = int(box.cls[0])
               class_name = classNames[cls]

               if class_name in screen_classes:
                   screen_region = img[y1:y2, x1:x2]
                   blurred_region = cv2.GaussianBlur(screen_region, (51, 51), 0)
                   img[y1:y2, x1:x2] = blurred_region

                   cvzone.cornerRect(img, (x1, y1, w, h))
                   conf = math.ceil((box.conf[0] * 100)) / 100
                   cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

       fps = 1 / (new_frame_time - prev_frame_time)
       prev_frame_time = new_frame_time

       cv2.imshow("Webcam Detection", img)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()
   ```

## Demonstration

![Demo](demo.gif)

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes.


## Contact

For any inquiries or support, please contact [musharrafhamraz02@gmail.com].

---

Enjoy the project and happy coding!