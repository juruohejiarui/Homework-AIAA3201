# Lab: Features & Matching

Welcome to the Computer Vision lab on Features and Matching! In this assignment, you will gain hands-on experience with extracting image features, matching them across different views, and utilizing these correspondences to stitch images together.

---

## ⚙️ Setup & Execution

### 1. Dependencies

Before you begin, ensure you have the required Python libraries installed in your environment. You can install them using `pip`:

```bash
pip install opencv-python numpy matplotlib
```

### 2. Project Structure

Your working directory should look like this:

```text
├── main.py               # The main script you will run to execute all tasks
├── harris_corner.py      # Script containing Task 1 logic
├── ratio_test.py         # Script containing Task 2 logic
├── alignment.py          # Script containing Task 3 logic
├── assets/               # Folder for all input images
│   ├── image1.png        # Input image for Task 1
│   ├── image2_a.png      # Input image A for Task 2
│   ├── image2_b.png      # Input image B for Task 2
│   ├── image3_a.png      # Input image A for Task 3
│   └── image3_b.png      # Input image B for Task 3
└── README.md             # This instruction document
```

### 3. How to Run

Open the respective Python files for each task (`harris_corner.py`, `ratio_test.py`, `alignment.py`). You will find several blocks of code marked with `TODO`. You must complete these sections to make the pipeline work. Once you have filled in the missing code, execute the main script from your terminal:

```bash
python main.py
```

All generated visualizations and result images will be automatically saved in the `output/` folder in your current directory. A complete run produces these ten files:

```text
output/
├── task1_harris_base.jpg
├── task1_harris_high_quality.jpg
├── task1_harris_large_distance.jpg
├── task2_ratio_0.6.jpg
├── task2_ratio_0.75.jpg
├── task2_ratio_0.9.jpg
├── task2_ratio_curve.png
├── task3_inlier_matches.jpg
├── task3_warped_img1.jpg
└── task3_final_overlay.jpg
```

---

## 📝 Tasks & Theoretical Background

This lab is divided into three main tasks. Below are the details on what you need to implement and the mathematical principles behind them.

### Task 1: Harris Corner Parameter Exploration

* **Target File**: `harris_corner.py`
* **Where to Edit**: Inside the `explore_harris_parameters` function.
* **Your Goal**: Complete the `cv2.goodFeaturesToTrack` API call. You need to map the provided `qualityLevel` and `minDistance` parameters to observe how they affect the detection results.



**💡 Principle:**
The Harris corner detector relies on the Structure Tensor $M$, which captures the gradient energy and correlation within a local window . To avoid the computationally expensive process of explicitly solving for eigenvalues, the Harris response function $R$ is calculated directly from the matrix elements:
$$R = \det(M) - k(\text{trace}(M))^2$$
This works because $\det(M) = \lambda_1\lambda_2$ and $\text{trace}(M) = \lambda_1+\lambda_2$.


* **`qualityLevel`**: Acts as a threshold. Increasing it drops the corner count sharply, keeping only the most prominent corners.


* **`minDistance`**: Applies Non-Maximum Suppression (NMS) spatially. It greedily selects candidates by their $R$ value while excluding other candidates within the `minDistance` radius, leading to evenly distributed corners.

---
### Task 2: Lowe's Ratio Test

* **Target File**: `ratio_test.py`
* **Where to Edit**: Inside the `experiment_ratio_test` function.
* **Your Goal**: Implement the logic for Lowe's Ratio Test inside the `for m, n in knn_matches:` loop to filter out bad matches.

**💡 Principle:**
A Brute-Force nearest-neighbor match often fails due to repetitive textures (like brick walls or grid windows) where many candidates look equally similar.

Lowe's Ratio Test solves this by finding the nearest neighbor $m_1$ (with distance $d_1$) and the second nearest neighbor $m_2$ (with distance $d_2$).

**The Condition**: Keep the match if $d_1/d_2<0.75$.

**Why it works**: For a good match, $d_1$ is small and the ratio is near $0$ . For repetitive textures, the distances are similar, making the ratio near $1$, so the match is discarded.

---

### Task 3: End-to-End Image Alignment (Advanced)

* **Target File**: `alignment.py`
* **Where to Edit**: Inside the `align_and_stitch` function.
* **Your Goal**: Extract the source and destination points from the `good` matches, compute the Homography matrix using RANSAC, and warp the first image to align with the second image.

**💡 Principle:**
Even after the Ratio Test, some false matches may remain due to locally similar patches . To fix this, we enforce a global geometric constraint: all correct matches must satisfy the same projective transform, known as the Homography Matrix $H$.

We use **RANSAC (Random Sample Consensus)** to robustly estimate . The algorithm repeats $N$ times:

1. Randomly sample $4$ non-collinear point pairs.


2. Solve for a candidate homography $H$.


3. Compute the reprojection error for all matches; if the error is below a threshold (e.g., $5$ px), it is counted as an inlier.


4. After $N$ iterations, keep the $H$ with the most inliers and re-fit it using all inlier points.

Once the accurate $H$ is found, we warp Image A into Image B's coordinate frame and blend them . If they align perfectly, your $H$ is correct.