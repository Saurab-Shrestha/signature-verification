from io import BytesIO
import os
from PIL import Image
from typing import Tuple, Union

import cv2
import numpy as np
import requests
from scipy import ndimage
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms


class SignatureCleaner:
    def __init__(self, device: str = None):
        """
        Initialize SignatureCleaner by loading the BiRefNet model.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading BiRefNet model...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        Load an image from path, URL, or PIL object.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                response = requests.get(image_input)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_input)
            return image.convert("RGB")
        else:
            raise ValueError("Image must be a file path, URL, or PIL Image object")

    def clean_image(self, image: Union[str, Image.Image], with_alpha: bool = False) -> Tuple[Image.Image, np.ndarray]:
        """
        Remove background using BiRefNet and return cleaned image.

        Args:
            image (str or PIL.Image): Input image path or object.
            with_alpha (bool): If True, return image with alpha (RGBA). Default is False (RGB).

        Returns:
            Tuple[Image.Image, np.ndarray]: Cleaned image and binary mask.
        """
        image = self.load_image(image)
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)[-1].sigmoid().cpu()

        mask = prediction[0].squeeze()
        mask_pil = transforms.ToPILImage()(mask)
        mask_resized = mask_pil.resize(original_size)

        if with_alpha:
            # Add alpha channel to original image
            result_image = image.copy().convert("RGBA")
            result_image.putalpha(mask_resized)
        else:
            # Composite over white background to get RGB
            image_rgb = image.convert('RGB')
            background = Image.new('RGB', image.size, (255, 255, 255))
            alpha_mask = mask_resized.convert("L")
            result_image = Image.composite(image_rgb, background, alpha_mask)

        return result_image, np.array(mask_resized)

    def normalize_image(self, img: np.ndarray, size: Tuple[int, int] = (840, 1360)) -> np.ndarray:
        """
        Normalize image size and position with dynamic canvas sizing.
        
        Args:
            img (np.ndarray): Input image as numpy array
            size (Tuple[int, int]): Maximum allowed size (height, width)
        
        Returns:
            np.ndarray: Normalized image
        """
        max_r_limit, max_c_limit = size

        # Apply gaussian filter to remove small components
        blur_radius = 0
        blurred_image = ndimage.gaussian_filter(img, blur_radius)

        # Binarize the image using OTSU's algorithm
        if blurred_image.dtype != np.uint8:
            blurred_image = (blurred_image / blurred_image.max() * 255).astype(np.uint8)

        threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the center of mass of the foreground pixels
        r, c = np.where(binarized_image == 0)

        # If no foreground pixels found, return blank canvas
        if r.size == 0 or c.size == 0:
            print("Warning: No foreground pixels found. Returning a blank canvas of max size.")
            return np.ones(size, dtype=np.uint8) * 255

        r_center = int(r.mean() - r.min())
        c_center = int(c.mean() - c.min())

        # Crop the image with tight bounding box
        cropped = img[r.min(): r.max(), c.min(): c.max()]

        # Dynamically determine output canvas size
        img_r, img_c = cropped.shape
        border_padding = 80

        # Calculate desired canvas size
        desired_r = min(max_r_limit, img_r + 2 * border_padding)
        desired_c = min(max_c_limit, img_c + 2 * border_padding)

        # Calculate starting positions to center the image
        r_start = (desired_r // 2) - r_center
        c_start = (desired_c // 2) - c_center

        # Adjust start positions to ensure they fit
        if r_start < 0:
            r_start = 0
        elif r_start + img_r > desired_r:
            r_start = desired_r - img_r
            if r_start < 0:
                r_start = 0
                if img_r > desired_r:
                    print(f"Warning: Image height ({img_r}) exceeds desired canvas height ({desired_r}). Content may be cropped.")
                    cropped = cropped[:desired_r, :]
                    img_r = desired_r

        if c_start < 0:
            c_start = 0
        elif c_start + img_c > desired_c:
            c_start = desired_c - img_c
            if c_start < 0:
                c_start = 0
                if img_c > desired_c:
                    print(f"Warning: Image width ({img_c}) exceeds desired canvas width ({desired_c}). Content may be cropped.")
                    cropped = cropped[:, :desired_c]
                    img_c = desired_c

        # Create normalized image with dynamic size
        normalized_image = np.ones((desired_r, desired_c), dtype=np.uint8) * 255

        # Add cropped and centered image to canvas
        normalized_image[r_start:r_start + img_r, c_start:c_start + img_c] = cropped

        # Remove noise - pixels above threshold set to white
        normalized_image[normalized_image > threshold] = 255

        return normalized_image

    def save_image(self, image: Image.Image, save_path: str):
        """
        Save the cleaned image to the given path.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # If the image has an alpha channel, flatten it
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            image = Image.alpha_composite(background.convert('RGBA'), image).convert('RGB')

        # Save the image
        image.save(save_path)
        print(f"Saved cleaned image to: {save_path}")

    def clean_and_normalize(self, image_input: Union[str, Image.Image], save_path: str, 
                                normalize_size: Tuple[int, int] = (840, 1360), show: bool = False):

        original = self.load_image(image_input)
        cleaned_image, _ = self.clean_image(original)
        cleaned_gray = np.array(cleaned_image.convert('L'))        
        normalized_image = self.normalize_image(cleaned_gray, normalize_size)
        normalized_pil = Image.fromarray(normalized_image, 'L')
        return original, cleaned_image, normalized_pil
    
    def clean_normalize_and_save(self, image_input: Union[str, Image.Image], save_path: str, 
                                normalize_size: Tuple[int, int] = (840, 1360), show: bool = False):
        """
        Clean image, normalize it, and save results. Optionally visualize the full process.
        
        Args:
            image_input: Input image path or PIL Image
            save_path: Path to save the final normalized image
            normalize_size: Target size for normalization (height, width)
            show: Whether to display visualization
        """
    
        original = self.load_image(image_input)
        cleaned_image, _ = self.clean_image(original)
        cleaned_gray = np.array(cleaned_image.convert('L'))
        normalized_image = self.normalize_image(cleaned_gray, normalize_size)
        normalized_pil = Image.fromarray(normalized_image, 'L')
        self.save_image(normalized_pil, save_path)
        if show:
            self.visualize_full_process(original, cleaned_image, normalized_image)
        
        return original, cleaned_image, normalized_pil
    
    def clean_and_save(self, image_input: Union[str, Image.Image], save_path: str, show: bool = False):
        """
        Clean image and save result to directory. Optionally visualize it.
        """
        original = self.load_image(image_input)
        cleaned_image, _ = self.clean_image(original)
        self.save_image(cleaned_image, save_path)

        if show:
            self.visualize(original, cleaned_image)
