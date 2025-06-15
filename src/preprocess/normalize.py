import cv2
import numpy as np
from scipy import ndimage

def normalize_image(img, size=(840, 1360)):
 
    max_r_limit, max_c_limit = size # Rename for clarity: these are now limits, not fixed output size

    # 1) Crop the image before getting the center of mass
    # Apply a gaussian filter on the image to remove small components
    blur_radius = 0
    blurred_image = ndimage.gaussian_filter(img, blur_radius)

    # Binarize the image using OTSU's algorithm.
    # cv2.THRESH_OTSU expects an 8-bit single-channel image.
    if blurred_image.dtype != np.uint8:
        blurred_image = (blurred_image / blurred_image.max() * 255).astype(np.uint8)

    threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find the center of mass of the foreground pixels (where binarized_image is 0)
    r, c = np.where(binarized_image == 0)

    # If no foreground pixels are found (e.g., completely white image),
    # return a blank canvas of the maximum allowed size to avoid errors.
    if r.size == 0 or c.size == 0:
        print("Warning: No foreground pixels found. Returning a blank canvas of max size.")
        return np.ones(size, dtype=np.uint8) * 255

    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    # Crop the image with a tight bounding box based on foreground pixels
    cropped = img[r.min(): r.max(), c.min(): c.max()]

    # 2) Dynamically determine the output canvas size to reduce padding
    img_r, img_c = cropped.shape
    
    # Define a minimal border around the cropped content
    # This border will be added on all sides (top, bottom, left, right)
    border_padding = 80 # Adjust this value to control the minimal padding

    # Calculate the desired height and width for the output canvas
    # This will be the cropped image size plus the border, but not exceeding the max_limit
    desired_r = min(max_r_limit, img_r + 2 * border_padding)
    desired_c = min(max_c_limit, img_c + 2 * border_padding)

    # Calculate the starting positions to center the cropped image within the desired canvas
    # If the desired canvas is exactly the size of the cropped image + border,
    # then r_start/c_start will be `border_padding`.
    # If the desired canvas is larger (due to max_limit being larger),
    # then it will center within that larger space.
    r_start = (desired_r // 2) - r_center
    c_start = (desired_c // 2) - c_center

    # Adjust start positions to ensure they are not negative and image fits
    if r_start < 0:
        r_start = 0
    elif r_start + img_r > desired_r: # If image extends beyond desired_r, re-center or crop
        r_start = desired_r - img_r
        if r_start < 0: # If still negative, it means img_r > desired_r (should be caught by initial cropping)
            r_start = 0
            # If the image is still too large after dynamic sizing and re-centering, it implies
            # img_r > desired_r. This case should ideally be handled by the initial cropping logic,
            # but as a fallback, ensure the cropped part doesn't exceed desired_r.
            # This might lead to further cropping if the image is still too big.
            if img_r > desired_r:
                print(f"Warning: Image height ({img_r}) still exceeds desired canvas height ({desired_r}). Content may be cropped.")
                cropped = cropped[:desired_r, :]
                img_r = desired_r


    if c_start < 0:
        c_start = 0
    elif c_start + img_c > desired_c:
        c_start = desired_c - img_c
        if c_start < 0:
            c_start = 0
            if img_c > desired_c:
                print(f"Warning: Image width ({img_c}) still exceeds desired canvas width ({desired_c}). Content may be cropped.")
                cropped = cropped[:, :desired_c]
                img_c = desired_c


    # Create the new normalized image with the dynamically determined tighter size
    normalized_image = np.ones((desired_r, desired_c), dtype=np.uint8) * 255

    # Add the cropped and centered image to the blank canvas
    normalized_image[r_start:r_start + img_r, c_start:c_start + img_c] = cropped

    # Remove noise - anything higher than the threshold. Pixels above the threshold are set to white.
    normalized_image[normalized_image > threshold] = 255

    return normalized_image
