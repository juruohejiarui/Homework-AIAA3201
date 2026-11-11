import os
import cv2

def ensure_dir(directory):
    """Ensure the output directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_image(image_path):
    """Load an image using OpenCV and convert to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot find image at {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)