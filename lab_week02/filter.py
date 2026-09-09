import numpy as np
import cv2

def my_imfilter(image, kernel):
    """
    Apply a 2D filter to an image using manual convolution.

    Args:
        image: A 2D (grayscale) or 3D (color) numpy array.
        kernel: A 2D numpy array representing the filter.

    Returns:
        output: Filtered image.
    """

    # Get image and kernel dimensions
    i_h, i_w = image.shape[:2]
    k_h, k_w = kernel.shape

    # Calculate padding size (assuming kernel is odd)
    pad_h = k_h // 2
    pad_w = k_w // 2

    # 1. Padding
    # We use 'reflect' padding as suggested in the lecture slides
    # to handle boundaries.
    if len(image.shape) == 2:
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        output = np.zeros_like(image)
    else:
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
        output = np.zeros_like(image)

    # 2. Convolution Loop
    # Slide the kernel over the image and compute the weighted sum (dot product).
    for y in range(i_h):
        for x in range(i_w):
            region = padded_image[y:y+k_h, x:x+k_w]
            if len(image.shape) == 2:
                output[y, x] = np.sum(region * kernel)
            else:
                output[y, x, :] = np.sum(region * kernel[:, :, None], axis=(0, 1))

    return np.clip(output, 0, 255).astype(image.dtype)
