import cv2
import os
import numpy as np
from filter import my_imfilter
from denoise import add_gaussian_noise, add_salt_and_pepper_noise, denoise_image
from hybrid import create_hybrid_image

# Setup paths
ASSETS_DIR = 'assets'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_part1():
    print("--- Running Part 1: Implementation of my_imfilter ---")
    img_path = os.path.join(ASSETS_DIR, 'image1.jpg')
    image = cv2.imread(img_path)
    if image is None: 
        print(f"Error: {img_path} not found.")
        return

    # Define an Identity filter (should assume output = input)
    identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    
    # Define a simple Box Blur filter (3x3)
    box_filter = np.ones((3, 3)) / 9.0

    filtered_identity = my_imfilter(image, identity_filter)
    filtered_box = my_imfilter(image, box_filter)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'part1_identity.jpg'), filtered_identity)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'part1_box_blur.jpg'), filtered_box)
    print("Part 1 Done. Check output folder.")

def run_part2():
    print("\n--- Running Part 2: Denoising Battle ---")
    img_path = os.path.join(ASSETS_DIR, 'image2.jpg')
    image = cv2.imread(img_path)
    if image is None: 
        print(f"Error: {img_path} not found.")
        return

    # --- Scenario A: Gaussian Noise ---
    print("Generating Gaussian Noise...")
    noisy_gauss = add_gaussian_noise(image, sigma=25)
    
    # Try both filters on Gaussian Noise
    # 1. Gaussian Filter on Gaussian Noise (Should work well)
    restored_gauss_by_gauss = denoise_image(noisy_gauss, 'gaussian', kernel_size=5)
    # 2. Median Filter on Gaussian Noise (Works, but may lose some detail or look 'painted')
    restored_gauss_by_median = denoise_image(noisy_gauss, 'median', kernel_size=5)

    # Save results
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'task2_A_noisy_gaussian.jpg'), noisy_gauss)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'task2_A_restore_by_gaussian.jpg'), restored_gauss_by_gauss)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'task2_A_restore_by_median.jpg'), restored_gauss_by_median)

    # --- Scenario B: Salt & Pepper Noise ---
    print("Generating Salt & Pepper Noise...")
    noisy_sp = add_salt_and_pepper_noise(image, prob=0.05)
    
    # Try both filters on S&P Noise
    # 1. Gaussian Filter on S&P (Result: blurry noise, not removed)
    restored_sp_by_gauss = denoise_image(noisy_sp, 'gaussian', kernel_size=5)
    # 2. Median Filter on S&P (Result: clean removal)
    restored_sp_by_median = denoise_image(noisy_sp, 'median', kernel_size=5)

    # Save results
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'task2_B_noisy_sp.jpg'), noisy_sp)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'task2_B_restore_by_gaussian.jpg'), restored_sp_by_gauss)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'task2_B_restore_by_median.jpg'), restored_sp_by_median)

    print("Part 2 Done. Check the 'output' folder for comparison.")
    print("-" * 40)
    print("OBSERVATION GUIDE (What you should see):")
    print("1. Gaussian Noise: Gaussian filter smoothes it out well. Median filter also works but might look 'cartoonish'.")
    print("2. Salt & Pepper Noise: Gaussian filter FAILS (it just blurs the white/black dots). Median filter SUCCEEDS perfectly.")
    print("-" * 40)

def run_part3():
    print("\n--- Running Part 3: Hybrid Images ---")
    # Assuming two images exist for hybrid demo
    # For testing, ensure image3a and image3b are the same size
    img_a = cv2.imread(os.path.join(ASSETS_DIR, 'image3a.jpg'))
    img_b = cv2.imread(os.path.join(ASSETS_DIR, 'image3b.jpg'))
    
    if img_a is None or img_b is None:
        print("Skipping Part 3: assets/image3a.jpg or image3b.jpg not found.")
        return
        
    # Resize to match
    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    # Create hybrid
    cutoff = 7 # Sigma value
    hybrid = create_hybrid_image(img_a, img_b, cutoff)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'part3_hybrid.jpg'), hybrid)
    print("Part 3 Done. Check output folder.")

if __name__ == '__main__':
    run_part1()
    run_part2()
    run_part3()