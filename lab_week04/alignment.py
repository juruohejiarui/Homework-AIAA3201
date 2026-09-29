import cv2
import numpy as np
import os

def align_and_stitch(img_path_a, img_path_b, output_dir):
    """
    Lab Task 3: Full pipeline from SIFT matching to RANSAC homography and image warping.
    """
    print(f"--- Running Task 3: Alignment & Stitching on {img_path_a} and {img_path_b} ---")
    img1 = cv2.imread(img_path_a)
    img2 = cv2.imread(img_path_b)
    
    if img1 is None or img2 is None:
        print("Error: Could not read input images for Task 3.")
        return

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        print("Not enough matches found to compute homography.")
        return

    # ==========================================
    # TODO: Extract the coordinates of the good matches
    # HINT: Ensure the shape is (-1, 1, 2) and type is np.float32 for findHomography
    # ==========================================
    
    src_pts = None # Replace
    dst_pts = None # Replace
    
    

    # ==========================================
    # TODO: Compute Homography using RANSAC
    # HINT: Use cv2.findHomography with cv2.RANSAC and a threshold (e.g., 5.0)
    # ==========================================

    H, mask = None, None # Replace
    
    
    
    if H is None:
        print("Homography computation failed.")
        return

    # Draw purely the inlier matches
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imwrite(os.path.join(output_dir, "task3_inlier_matches.jpg"), img_inliers)

    # ==========================================
    # TODO: Warp Image 1 to Image 2's coordinate frame and blend them
    # HINT: Use cv2.warpPerspective and cv2.addWeighted
    # ==========================================
    h, w = img2.shape[:2]
    img1_warped = None # Replace
    overlay = None # Replace
    
    cv2.imwrite(os.path.join(output_dir, "task3_warped_img1.jpg"), img1_warped)
    cv2.imwrite(os.path.join(output_dir, "task3_final_overlay.jpg"), overlay)
    print(f"Alignment complete. Results saved in {output_dir}")