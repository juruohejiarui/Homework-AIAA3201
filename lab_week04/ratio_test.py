import cv2
import os
import matplotlib.pyplot as plt

def experiment_ratio_test(img_path_a, img_path_b, output_dir):
    """
    Lab Task 2: Implement Lowe's Ratio Test and observe how the threshold affects matches.
    """
    print(f"--- Running Task 2: Lowe's Ratio Test on {img_path_a} and {img_path_b} ---")
    img1 = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: Could not read input images for Task 2.")
        return

    # Initialize SIFT detector and find keypoints/descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Get 2 nearest neighbors for the ratio test
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(des1, des2, k=2)
    
    thresholds = [0.60, 0.75, 0.90]
    match_counts = []
    
    for thresh in thresholds:
        good_matches = []
        # ==========================================
        # TODO: Implement Lowe's Ratio Test
        # HINT: For each pair of matches (m, n), check if m.distance < thresh * n.distance
        # ==========================================
        for m, n in knn_matches:
            pass # Replace this with your implementation
                
                
        match_counts.append(len(good_matches))
        
        # Draw matches for visual inspection
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        out_name = os.path.join(output_dir, f"task2_ratio_{thresh}.jpg")
        cv2.imwrite(out_name, img_matches)
        print(f"Threshold: {thresh} -> Good Matches: {len(good_matches)}")

    # Plot the threshold vs matches curve
    plt.figure()
    plt.plot(thresholds, match_counts, marker='o', linestyle='-')
    plt.title("Ratio Threshold vs # Good Matches")
    plt.xlabel("Ratio Threshold")
    plt.ylabel("# Good Matches")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "task2_ratio_curve.png"))
    plt.close()