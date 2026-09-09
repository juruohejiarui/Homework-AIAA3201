#!/usr/bin/env python3
"""
Week 01 Lab Main Program
Reads images, applies various Instagram-style filters, generates effect montages
"""

import cv2
import numpy as np
import os
from filters import get_filter_info


def create_filter_montage(original_image, filter_results):
    """
    Create filter effect montage
    Directly concatenate original image and 5 filter effect images in 2x3 layout

    Args:
        original_image: Original image
        filter_results: Filter results dictionary {filter_name: processed_image}

    Returns:
        Montage image
    """
    # Get image dimensions (use first image dimensions as reference)
    h, w = original_image.shape[:2]

    # Prepare image list
    images = [original_image]  # Original image

    # Add filter effects
    for result_img in filter_results.values():
        images.append(result_img)

    # Create 2x3 montage (total 6 images)
    rows, cols = 2, 3

    # Create montage canvas
    montage = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    # Place images
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols

        # Calculate position
        y_start = row * h
        x_start = col * w

        # Place image
        montage[y_start:y_start+h, x_start:x_start+w] = img

    return montage


def process_image(image_path, output_dir='output'):
    """
    Process single image, apply all filters and generate montage and individual filter images

    Args:
        image_path: Image path
        output_dir: Output directory

    Returns:
        Whether processing was successful
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found {image_path}")
            return False

        # Read image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Cannot read image {image_path}")
            return False

        print(f"Processing image: {os.path.basename(image_path)}")

        # Get base filename and create subdirectory
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)

        # Get filter information
        filter_info = get_filter_info()

        # Apply all filters
        filter_results = {}
        for filter_key, info in filter_info.items():
            print(f"  Applying filter: {info['name']}")
            result = info['function'](original_image)
            filter_results[filter_key] = result

            # Save individual filter image
            filter_filename = f"{filter_key}.jpg"
            filter_path = os.path.join(image_output_dir, filter_filename)
            cv2.imwrite(filter_path, result)
            print(f"    Saved: {filter_filename}")

        # Generate montage
        print("Generating effect montage...")
        montage = create_filter_montage(original_image, filter_results)

        # Save montage
        output_path = os.path.join(image_output_dir, "montage.jpg")
        cv2.imwrite(output_path, montage)

        print(f"Montage saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False


def main():
    """
    Main function
    """
    print("=== Week 01 Lab: Instagram Style Filter Experience ===\n")

    # Check assets directory
    assets_dir = 'assets'
    if not os.path.exists(assets_dir):
        print(f"Error: Assets directory not found {assets_dir}")
        return

    # Get all image files
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for file in os.listdir(assets_dir):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(os.path.join(assets_dir, file))

    if not image_files:
        print(f"No supported image files found in {assets_dir} directory")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return

    print(f"Found {len(image_files)} images, starting processing...\n")

    # Process each image
    success_count = 0
    for image_path in image_files:
        if process_image(image_path):
            success_count += 1
        print()  # Empty line separator

    print(f"Processing completed! Successfully processed {success_count}/{len(image_files)} images")

    # Display results
    if success_count > 0:
        print("\nResults saved in 'output' directory")
        print("- Individual filter images: output/{image_name}/{filter_name}.jpg")
        print("- Effect montage: output/{image_name}/montage.jpg")
        print("Use your system image viewer to check the results")


if __name__ == "__main__":
    main()
