import os
import cv2
import requests
import base64
from io import BytesIO
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Dict, List, Any


class EndpointHandler:
    def __init__(self, path: str):
        """Initialize the EndpointHandler with the model path.

        Args:
            path (str): Path to the directory containing the model files
        """
        # Model info
        self.repo_id = "tech4humans/yolov8s-signature-detector"
        self.filename = "model.onnx"
        self.model_dir = path
        self.model_path = os.path.join(self.model_dir, "model.onnx")

        # Model parameters
        self.classes = ["signature"]
        self.input_width = 640
        self.input_height = 640

        # Initialize ONNX Runtime session
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(self.model_path, options)
        self.session.set_providers(
            ["OpenVINOExecutionProvider"], [{"device_type": "CPU"}]
        )

        # Initialize color palette for visualization
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def _load_image(self, image_input: str) -> Image.Image:
        """Load image from various input formats.

        Args:
            image_input (str): Can be:
                - URL to an image
                - Base64 encoded image string
                - Path to local image file

        Returns:
            PIL.Image: Loaded image
        """
        # Check if input is a URL
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input)
            response.raise_for_status()  # Raise exception for bad status codes
            return Image.open(BytesIO(response.content))

        # Check if input might be base64 encoded
        if ";base64," in image_input:
            # Extract the actual base64 string after the comma
            base64_string = image_input.split(";base64,")[1]
            img_data = base64.b64decode(base64_string)
            return Image.open(BytesIO(img_data))
        elif len(image_input) % 4 == 0:  # Possible base64 without header
            try:
                img_data = base64.b64decode(image_input)
                return Image.open(BytesIO(img_data))
            except:
                pass

        # Try to open as local file
        return Image.open(image_input)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request to detect signatures in an image.

        Args:
            data (Dict[str, Any]): Request data containing:
                - image (str): Base64 encoded image or image file path
                - confidence_threshold (float, optional): Detection confidence threshold
                - iou_threshold (float, optional): NMS IoU threshold

        Returns:
            Dict[str, Any]: Detection results containing:
                - image: Base64 encoded image with detections drawn
                - detections: List of detected signatures with coordinates and confidence
        """
        # Get parameters from request
        conf_thres = float(data.get("confidence_threshold", 0.25))
        iou_thres = float(data.get("iou_threshold", 0.45))

        # Handle different input image formats
        if "image" not in data:
            raise ValueError("No image provided in request")

        try:
            pil_image = self._load_image(data["image"])
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

        # Run detection
        img_data, original_image = self._preprocess(pil_image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})

        # Process detections
        detections = self._process_detections(
            original_image, outputs, conf_thres, iou_thres
        )

        # Draw detections on image
        output_image = self._draw_detections(original_image, detections)

        # Convert output image to base64
        buffered = BytesIO()
        Image.fromarray(output_image).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "image": img_str,
            "detections": [
                {
                    "bbox": detection["box"],
                    "score": float(detection["score"]),
                    "class": self.classes[detection["class_id"]],
                }
                for detection in detections
            ],
        }

    def _preprocess(self, img: Image.Image) -> tuple:
        """Preprocess the input image for inference.

        Args:
            img (PIL.Image): Input image

        Returns:
            tuple: Preprocessed image data and original cv2 image
        """
        # Convert PIL Image to cv2 format
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Store original dimensions
        self.img_height, self.img_width = img_cv2.shape[:2]

        # Preprocess for model
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))

        # Normalize and transpose
        image_data = np.array(img_resized) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data, img_cv2

    def _process_detections(
        self, input_image: np.ndarray, output: List, conf_thres: float, iou_thres: float
    ) -> List[Dict]:
        """Process model outputs to get detections.

        Args:
            input_image (np.ndarray): Original image
            output (List): Model output
            conf_thres (float): Confidence threshold
            iou_thres (float): IoU threshold for NMS

        Returns:
            List[Dict]: List of processed detections
        """
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= conf_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0:4]

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

        detections = []
        for i in indices:
            detections.append(
                {"box": boxes[i], "score": scores[i], "class_id": class_ids[i]}
            )

        return detections

    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection boxes and labels on the image.

        Args:
            image (np.ndarray): Input image
            detections (List[Dict]): List of detections

        Returns:
            np.ndarray: Image with drawn detections
        """
        img_copy = image.copy()

        for det in detections:
            box = det["box"]
            score = det["score"]
            class_id = det["class_id"]

            x1, y1, w, h = box
            color = self.color_palette[class_id]

            # Draw box
            cv2.rectangle(
                img_copy, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2
            )

            # Draw label
            label = f"{self.classes[class_id]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            cv2.rectangle(
                img_copy,
                (int(label_x), int(label_y - label_height)),
                (int(label_x + label_width), int(label_y + label_height)),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                img_copy,
                label,
                (int(label_x), int(label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
