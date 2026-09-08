"""
Week 01 Lab: Instagram Style Filter Implementation
Contains 5 classic filter effect functions
"""

import cv2
import numpy as np


def apply_vintage_warm(image):
    """
    Vintage warm tone filter
    Adds yellow-orange tones, simulates film texture
    Implementation: RGB channel weighted adjustment

    Args:
        image: Input image (BGR format)

    Returns:
        Processed image
    """
    # Create copy to avoid modifying original image
    result = image.copy().astype(np.float32)

    # RGB channel weighted adjustment, enhance red and green channels for warm tone
    # R channel enhancement (warm tone)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)  # R channel * 1.1
    # G channel slight enhancement
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.05, 0, 255)  # G channel * 1.05
    # B channel reduction (reduce cool tones)
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.95, 0, 255)  # B channel * 0.95

    # Increase overall brightness for film texture
    result = np.clip(result + 10, 0, 255)

    return result.astype(np.uint8)


def apply_cool_tone(image):
    """
    Cool tone filter
    Adds blue tones, creates cool atmosphere
    Implementation: Enhance B channel, reduce R/G channels

    Args:
        image: Input image (BGR format)

    Returns:
        Processed image
    """
    result = image.copy().astype(np.float32)

    # Enhance blue channel (cool tone)
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.2, 0, 255)  # B channel * 1.2
    # Reduce red channel
    result[:, :, 2] = np.clip(result[:, :, 2] * 0.9, 0, 255)  # R channel * 0.9
    # Reduce green channel
    result[:, :, 1] = np.clip(result[:, :, 1] * 0.95, 0, 255)  # G channel * 0.95

    # Slightly reduce brightness to enhance cool atmosphere
    result = np.clip(result - 5, 0, 255)

    return result.astype(np.uint8)


def apply_high_contrast(image):
    """
    High contrast filter
    Enhances light-dark contrast for more impact
    Implementation: Stretches histogram, increases dynamic range

    Args:
        image: Input image (BGR format)

    Returns:
        Processed image
    """
    # Convert to LAB color space for processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply histogram equalization to luminance channel
    l_eq = cv2.equalizeHist(l)

    # Merge channels
    lab_eq = cv2.merge([l_eq, a, b])

    # Convert back to BGR
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return result


def apply_faded(image):
    """
    Faded effect filter
    Reduces contrast and saturation for nostalgic feel
    Implementation: Compresses dynamic range, reduces saturation

    Args:
        image: Input image (BGR format)

    Returns:
        Processed image
    """
    result = image.copy().astype(np.float32)

    # Reduce contrast - shift pixel values toward 128
    result = result * 0.8 + 128 * 0.2

    # Reduce saturation - process in HSV color space
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Reduce saturation
    s = (s * 0.6).astype(np.uint8)

    # Merge and convert back to BGR
    hsv_faded = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_faded, cv2.COLOR_HSV2BGR)

    return result.astype(np.uint8)


def apply_grayscale(image):
    """
    Grayscale filter
    Converts to black and white, classic effect
    Implementation: RGB → Grayscale color space conversion

    Args:
        image: Input image (BGR format)

    Returns:
        Processed image
    """
    # Use OpenCV's standard grayscale conversion
    # Formula: Gray = 0.299*R + 0.587*G + 0.114*B
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert back to 3 channels for consistency
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return result


def get_filter_info():
    """
    Get information for all filters
    Used for display and selection

    Returns:
        Filter information dictionary
    """
    return {
        'vintage_warm': {
            'name': 'Vintage Warm',
            'function': apply_vintage_warm
        },
        'cool_tone': {
            'name': 'Cool Tone',
            'function': apply_cool_tone
        },
        'high_contrast': {
            'name': 'High Contrast',
            'function': apply_high_contrast
        },
        'faded': {
            'name': 'Faded',
            'function': apply_faded
        },
        'grayscale': {
            'name': 'Grayscale',
            'function': apply_grayscale
        }
    }
