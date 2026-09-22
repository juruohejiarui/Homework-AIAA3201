import cv2
import numpy as np

def match_image_size(image1, image2):
    """
    Utility function: Check if image2 has the same dimensions as image1.
    If not, resize image2 to match image1.

    Args:
        image1: The reference image.
        image2: The image to be resized.

    Returns:
        resized_image2: image2 with the same (height, width) as image1.
    """
    # Get dimensions (Height, Width)
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Check if dimensions match
    if (h1, w1) != (h2, w2):
        print(f"Warning: Resizing image2 from {(h2, w2)} to {(h1, w1)} to match image1.")
        # Note: cv2.resize expects (width, height), which is opposite of shape (height, width)
        return cv2.resize(image2, (w1, h1))

    return image2

def create_hybrid_image(image1, image2, cutoff_frequency):
    """
    Create a hybrid image by combining low frequencies of image1
    and high frequencies of image2.
    """

    # 0. Pre-processing: Ensure sizes match
    image2 = match_image_size(image1, image2)

    # Kernel size usually needs to be large enough to cover significant gaussian values
    # Rule of thumb from slides: kernel half-width >= 3 * sigma
    k_size = int(cutoff_frequency * 4) * 2 + 1

    # 1. Get Low Frequency of Image 1
    # TODO: Students implement Low-pass filtering
    # Hint: Use cv2.GaussianBlur(image, ksize=(width, height), sigmaX=value)
    #       k_size is already calculated for you above, and the cutoff
    #       frequency is the sigma value.
    low_frequencies = cv2.GaussianBlur(
        image1.astype(np.float64), (k_size, k_size), cutoff_frequency
    )

    # 2. Get High Frequency of Image 2
    # High Freq = Original - Low Freq (Smoothed)
    image2_f = image2.astype(np.float64)
    low_freq_2 = cv2.GaussianBlur(image2_f, (k_size, k_size), cutoff_frequency)
    high_frequencies = image2_f - low_freq_2

    # 3. Combine them
    hybrid_image = low_frequencies + high_frequencies

    # Clip values to be in valid range [0, 255] just in case
    return np.clip(hybrid_image, 0, 255).astype(np.uint8)
