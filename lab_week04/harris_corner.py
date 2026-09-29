import cv2
import numpy as np
import os

def explore_harris_parameters(img_path, output_dir):
    """
    Lab Task 1: Explore the effect of qualityLevel and minDistance on Harris Corners.
    """
    print(f"--- Running Task 1: Harris Corner Exploration on {img_path} ---")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read {img_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # We test scenarios based on the lecture slides:
    # 1. Base case
    # 2. High qualityLevel (fewer, stronger corners)
    # 3. Large minDistance (sparse, evenly distributed corners)
    scenarios = [
        {"qualityLevel": 0.01, "minDistance": 1, "name": "base"},
        {"qualityLevel": 0.2, "minDistance": 1, "name": "high_quality"},
        {"qualityLevel": 0.01, "minDistance": 20, "name": "large_distance"}
    ]
    
    for params in scenarios:
        # ==========================================
        # TODO: Implement corner detection using cv2.goodFeaturesToTrack
        # HINT: Pass the parameters from the 'scenarios' dictionary.
        # ==========================================
        
        corners = None # Replace this line with your implementation
        
        img_draw = img.copy()
        if corners is not None:
            corners = corners.astype(int)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(img_draw, (x, y), 3, (0, 0, 255), -1)
                
        out_name = os.path.join(output_dir, f"task1_harris_{params['name']}.jpg")
        cv2.imwrite(out_name, img_draw)
        print(f"Saved {out_name} with {len(corners) if corners is not None else 0} corners.")