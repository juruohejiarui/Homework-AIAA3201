import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=30):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.1):
    noisy = np.copy(image)
    # Salt mode
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    # Pepper mode
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

def denoise_image(noisy_image, filter_type, kernel_size=3):
    """
    Denoise the image using specific filter types.

    Args:
        noisy_image: Input noisy image.
        filter_type: String, either 'gaussian' or 'median'.
        kernel_size: Integer, size of the kernel (e.g., 3, 5).

    Returns:
        Denoised image.
    """
    # TODO: Students need to fill in the API calls here
    if filter_type == 'gaussian':
        # Apply Gaussian Blur
        # Hint: cv2.GaussianBlur(src, ksize, sigmaX)
        #       Set sigmaX to 0 so it is calculated from the kernel size.
        pass  # TODO: return the blurred image

    elif filter_type == 'median':
        # Apply Median Blur
        # Hint: cv2.medianBlur(src, ksize)
        #       Effective for removing salt-and-pepper noise.
        pass  # TODO: return the denoised image

    else:
        raise ValueError("Unsupported filter type")
