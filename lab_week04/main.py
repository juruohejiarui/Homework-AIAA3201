import os
from harris_corner import explore_harris_parameters
from ratio_test import experiment_ratio_test
from alignment import align_and_stitch

def ensure_dirs():
    """Ensure that the necessary directories exist."""
    assets_dir = "assets"
    output_dir = "output"
    
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"Created '{assets_dir}' directory. Please place your images here.")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    return assets_dir, output_dir

if __name__ == "__main__":
    assets_dir, output_dir = ensure_dirs()
    
    # Define input paths based on the assets directory
    img_q1 = os.path.join(assets_dir, "image1.png")
    img_q2_a = os.path.join(assets_dir, "image2_a.png")
    img_q2_b = os.path.join(assets_dir, "image2_b.png")
    img_q3_a = os.path.join(assets_dir, "image3_a.png")
    img_q3_b = os.path.join(assets_dir, "image3_b.png")
    
    # Check if the required files exist before running
    # If a student hasn't added the images yet, we catch it gracefully.

    if os.path.exists(img_q1):
        explore_harris_parameters(img_q1, output_dir)
    else:
        print(f"Skipping Task 1: {img_q1} not found in '{assets_dir}'.")

    if os.path.exists(img_q2_a) and os.path.exists(img_q2_b):
        experiment_ratio_test(img_q2_a, img_q2_b, output_dir)
    else:
        print(f"Skipping Task 2: Missing images for Task 2 in '{assets_dir}'.")

    if os.path.exists(img_q3_a) and os.path.exists(img_q3_b):
        align_and_stitch(img_q3_a, img_q3_b, output_dir)
    else:
        print(f"Skipping Task 3: Missing images for Task 3 in '{assets_dir}'.")

    print("\nLab execution finished. Check the 'output' directory for results.")