import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np

class BiRefNetProcessor:
    def __init__(self):
        """Initialize BiRefNet model and preprocessing pipeline"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision(["high", "highest"][0])
        
        # Load BiRefNet model
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def remove_background(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:
        original_size = image.size        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)[-1].sigmoid().cpu()
        
        mask = predictions[0].squeeze()
        mask_pil = transforms.ToPILImage()(mask)
        mask_resized = mask_pil.resize(original_size)
        
        result_image = image.copy()
        result_image.putalpha(mask_resized)        
        mask_np = np.array(mask_resized)
        
        return result_image, mask_np
    
    def process_image(self, image_path: str) -> tuple[Image.Image, np.ndarray]:
        image = Image.open(image_path).convert('RGB')
        return self.remove_background(image)

# Create a singleton instance
birefnet_processor = BiRefNetProcessor() 