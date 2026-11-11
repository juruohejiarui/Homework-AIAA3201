import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import utilities and tasks
from utils import ensure_dir, load_image
from task1_detection import run_object_detection, visualize_task1
from task2_segmentation import run_semantic_segmentation, visualize_task2
from task3_sam import run_sam_zeroshot, visualize_task3

def create_dummy_image(path, label="Dummy"):
    """Create a dummy image if real one is missing."""
    dummy_img = np.ones((512, 512, 3), dtype=np.uint8) * 200
    cv2.putText(dummy_img, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 3)
    cv2.circle(dummy_img, (256, 256), 100, (100, 150, 200), -1)
    plt.imsave(path, dummy_img)

def main():
    input_dir = "input"
    output_dir = "output"
    ensure_dir(input_dir)
    ensure_dir(output_dir)
    
    img_path_task1 = os.path.join(input_dir, "single_object_detection.png")
    img_path_task2 = os.path.join(input_dir, "semantic_segmentation.png")
    img_path_task3 = os.path.join(input_dir, "SAM_simulation.png")
    
    if not os.path.exists(img_path_task1): create_dummy_image(img_path_task1, "Missing Cat")
    if not os.path.exists(img_path_task2): create_dummy_image(img_path_task2, "Missing Street")
    if not os.path.exists(img_path_task3): create_dummy_image(img_path_task3, "Missing Mug")

    try:
        print("--- Running Lab Assignments (Real Inference) ---")
        
        # --- Task 1 Execution ---
        print("\n=> Starting Task 1: Object Detection")
        img1 = load_image(img_path_task1)
        boxes, labels, scores = run_object_detection(img1)
        task1_out = os.path.join(output_dir, "task1_detection_result.jpg")
        visualize_task1(img1, boxes, labels, scores, task1_out)
        
        # --- Task 2 Execution ---
        print("\n=> Starting Task 2: Semantic Segmentation")
        img2 = load_image(img_path_task2)
        mask_pred = run_semantic_segmentation(img2)
        task2_out = os.path.join(output_dir, "task2_segmentation_result.jpg")
        visualize_task2(img2, mask_pred, task2_out)
        
        # --- Task 3 Execution ---
        print("\n=> Starting Task 3: SAM Zero-Shot Transfer")
        img3 = load_image(img_path_task3)
        boxes3, labels3, scores3 = run_object_detection(img3)
        if len(boxes3) > 0:
            dynamic_bbox_prompt = boxes3[0]
            print(f"[Pipeline] Step 2: Found object with score {scores3[0]:.2f}. Passing BBox to SAM...")
            
            # Run SAM using the dynamic prompt generated just now
            best_mask = run_sam_zeroshot(img3, dynamic_bbox_prompt)
            
            task3_out = os.path.join(output_dir, "task3_sam_real_result.jpg")
            visualize_task3(img3, dynamic_bbox_prompt, best_mask, task3_out)
        else:
            print("[Pipeline] Error: No object detected in Task 3 image. Cannot run SAM.")
        
        print("\n🎉 All tasks completed successfully! Check the 'output' directory.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")

if __name__ == "__main__":
    main()