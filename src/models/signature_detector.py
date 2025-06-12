import os
from PIL import Image
import numpy as np
import torch
import cv2

# YOLOv5 imports
try:
    import sys
    from pathlib import Path
    YOLO_PATH = Path(__file__).parent.parent.parent / 'venv' / 'lib' / 'python3.11' / 'site-packages' / 'yolov5'
    if str(YOLO_PATH) not in sys.path:
        sys.path.append(str(YOLO_PATH))
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.dataloaders import LoadImages
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.torch_utils import select_device
except ImportError:
    print("Installing YOLOv5...")
    os.system("pip install yolov5")
    import sys
    from pathlib import Path
    YOLO_PATH = Path(__file__).parent.parent.parent / 'venv' / 'lib' / 'python3.11' / 'site-packages' / 'yolov5'
    if str(YOLO_PATH) not in sys.path:
        sys.path.append(str(YOLO_PATH))
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.dataloaders import LoadImages
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.torch_utils import select_device

# ---------------------
# ---------------------------
# YOLOv5 Signature Detection
# ---------------------------
class SignatureDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize YOLOv5 signature detector
        Args:
            model_path: Path to custom YOLOv5 model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.device = select_device('')
        
        if model_path and os.path.exists(model_path):
            # Load custom trained model
            self.model = DetectMultiBackend(model_path, device=self.device)
        else:
            # Load pre-trained YOLOv5 model (will be fine-tuned for signatures)
            self.model = DetectMultiBackend('yolov5s.pt', device=self.device)
            print("Warning: Using general YOLOv5 model. For best results, use a signature-specific model.")
        
        self.model.conf = confidence_threshold
        self.stride = self.model.stride
        self.imgsz = check_img_size((640, 640), s=self.stride)
        
    def detect_signatures(self, image_path):
        """
        Detect signatures in an image
        Args:
            image_path: Path to input image
        Returns:
            List of detection results with bounding boxes
        """
        # Load image
        dataset = LoadImages(image_path, img_size=self.imgsz, stride=self.stride)
        
        # Run inference
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            
            # Inference
            pred = self.model(im)
            
            # NMS
            pred = non_max_suppression(pred, self.confidence_threshold, 0.45)
            
            # Process predictions
            signature_detections = []
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        signature_detections.append({
                            'bbox': [int(x) for x in xyxy],
                            'confidence': float(conf),
                            'class': int(cls)
                        })
            
            return signature_detections
    
    def extract_signature_regions(self, image_path, padding=10):
        """
        Extract signature regions from image based on YOLO detections
        Args:
            image_path: Path to input image
            padding: Padding around detected bounding box
        Returns:
            List of PIL Images containing signature regions
        """
        # Load original image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Get detections
        detections = self.detect_signatures(image_path)
        
        signature_regions = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)
            
            # Extract region
            signature_region = image.crop((x1, y1, x2, y2))
            signature_regions.append({
                'image': signature_region,
                'bbox': [x1, y1, x2, y2],
                'confidence': detection['confidence'],
                'index': i
            })
        
        return signature_regions