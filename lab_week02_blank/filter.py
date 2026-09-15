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
    # TODO: Students need to write this line
    # Hint: np.pad(array, pad_width, mode='reflect')
    #       For a color image, pad the height and width axes only,
    #       not the channel axis.
    if len(image.shape) == 2:
        padded_image = None  # TODO: pad the grayscale image
        output = np.zeros_like(image)
    else:
        padded_image = None  # TODO: pad the color image
        output = np.zeros_like(image)

    # 2. Convolution Loop
    # Slide the kernel over the image and compute the weighted sum (dot product).
    # TODO: Students need to implement this double loop
    for y in range(i_h):
        for x in range(i_w):
            # TODO: For each output pixel, take the (k_h x k_w) region of
            #       padded_image that starts at (y, x), multiply it
            #       element-wise with the kernel, and store the sum in output.
            #       For color images, do this for every channel independently.
            pass

    return output
