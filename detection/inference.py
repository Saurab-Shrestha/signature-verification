import os
import cv2
import supervision as sv
from ultralytics import YOLO

DATASET_DIR = "/home/kshitiz/Documents/yolov8s-signature-detector/images"

FILENAME = "yolov8s.pt"  # Or ".onnx or .pt"
LOCAL_MODEL_DIR = "." 
MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, FILENAME)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def infer_single_image(model, image_path):
    """
    Perform inference on a single image.
    Args:
        model: YOLO model instance
        image_path: Path to the input image
    Returns:
        Annotated image with detections
    """
    image = cv2.imread(image_path)
    results = model(image)
    detections = sv.Detections.from_ultralytics(results[0])
    box_annotator = sv.BoxAnnotator()  # for visualization
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    return annotated_image


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    print(f"Loading model from: {MODEL_PATH}")
    test_images = [
        os.path.join(DATASET_DIR, fname)
        for fname in os.listdir(DATASET_DIR)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not test_images:
        raise FileNotFoundError(f"No images found in {DATASET_DIR}")
    

    # load model
    model = YOLO(MODEL_PATH)
    box_annotator = sv.BoxAnnotator() # for visualization

    for image_path in test_images:
        image_name = os.path.basename(image_path)
        print(f"Processing: {image_name}")
        annotated_image = infer_single_image(model, image_path)
        # Save annotated result
        output_path = os.path.join(OUTPUT_DIR, f"annotated_{image_name}")
        cv2.imwrite(output_path, annotated_image)

    print(f"All annotated images saved in: {OUTPUT_DIR}")
