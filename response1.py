# import  libraries
import cv2
from ultralytics import YOLO

# Load YOLOv8 model. This will download weights if not already present.
model = YOLO("yolov8n.pt")

# Define a function to detect objects in an image
def detect_objects(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Run the model on the image
    results = model(image)

    # Get the first detection result and draw bounding boxes
    detected_image = results[0].plot()

    # Show the image in a new window
    cv2.imshow("YOLOv8 Detection", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the detected objects
    print("Detected objects:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        print(f" - {label} ({confidence:.2f}) at {coords}")

# Change the file name below to your actual image
detect_objects("test_image.jpg")
