# Import libraries
import cv2
from ultralytics import YOLO

# Load the YOLOv8 nano model
# This will automatically download the model weights if not available
model = YOLO("yolov8n.pt")

# Define a function to detect objects in an image
def detect_objects(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Run object detection on the image
    results = model(image)

    # Draw bounding boxes on the image
    output_image = results[0].plot()

    # Show the image with detections
    cv2.imshow("Detected Objects", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display detection results in terminal
    print("Objects detected:")
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()
        print(f" - {class_name}: {confidence:.2f}, Coordinates: {coords}")

# Call the function with your image file
detect_objects("test_image.jpg")
