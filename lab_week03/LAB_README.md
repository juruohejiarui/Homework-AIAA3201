# Week 03 Lab Assignment: Document Scanner Implementation

## Learning Objectives

Implement a complete document scanning pipeline that can automatically locate documents from cluttered backgrounds and perform perspective correction.

## Background

Document scanning is a classic application in computer vision, combining core concepts of image processing and geometric transformations. Through this lab, you will learn:

- Image preprocessing techniques (grayscale conversion, Gaussian blur)
- Edge detection algorithms (Canny algorithm)
- Contour analysis and shape recognition
- Perspective transformation and geometric correction

## Environment

Python 3.8 or higher, with OpenCV, NumPy and Matplotlib.

```bash
pip install opencv-python numpy matplotlib
```

The tutorial notebook `edge_detection_tutorial.ipynb` in this folder is the demo from the lab session, kept here as a reference.

## Lab Tasks

### Task Overview

You need to complete four key TODO sections in `student_document_scanner.py` to implement the complete document scanning functionality.

### Specific Tasks

#### 1. Preprocessing Stage - Gaussian Blur Denoising (preprocess_image method)

**Location**: `student_document_scanner.py` line 46

**Task Description**:
- Use `cv2.GaussianBlur()` to implement Gaussian blur
- Choose appropriate kernel size and sigma values

**Discussion Questions**:
- Why is sigma selection crucial for edge detection?
- What are the effects of sigma being too small or too large?

**Hint**:
```python
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
```

#### 2. Edge Detection Stage - Canny Algorithm (detect_edges method)

**Location**: `student_document_scanner.py` line 64

**Task Description**:
- Use `cv2.Canny()` to implement edge detection
- Adjust appropriate threshold parameters

**Discussion Questions**:
- How does Canny algorithm's dual-threshold strategy work?
- How to balance detection accuracy and noise suppression?

**Hint**:
```python
edges = cv2.Canny(image, threshold1, threshold2)
```

#### 3. Contour Filtering Logic (find_document_contour method)

**Location**: `student_document_scanner.py` lines 88-99

**Task Description**:
- Implement the filtering logic in the contour traversal loop
- Combine area filtering and polygon approximation

**Specific Requirements**:
1. Calculate contour area: `cv2.contourArea(contour)`
2. Calculate contour perimeter: `cv2.arcLength(contour, True)`
3. Polygon approximation: `cv2.approxPolyDP(contour, epsilon, True)`
4. Filter quadrilaterals: `len(approx) == 4`
5. Select the largest valid contour by area

**Hint**:
```python
for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area_threshold:
        continue

    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter  # Adjust epsilon factor
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4 and area > max_area:
        # Update best contour
        pass
```

#### 4. Perspective Transformation Implementation (perspective_correction method)

**Location**: `student_document_scanner.py` lines 179-182

**Task Description**:
- Calculate perspective transformation matrix
- Apply perspective transformation

**Specific Requirements**:
1. Use `cv2.getPerspectiveTransform()` to calculate transformation matrix
2. Use `cv2.warpPerspective()` to apply transformation

**Hint**:
```python
M = cv2.getPerspectiveTransform(source_points, destination_points)
warped = cv2.warpPerspective(image, M, (width, height))
```

## Testing and Validation

### Running Tests

```bash
python student_document_scanner.py
```

### Test Cases

The lab provides three test cases:
- `asset/case1.png`: Standard document scanning scenario
- `asset/case2.png`: Complex background interference scenario
- `asset/case3.png`: Severely tilted document scenario

### Success Criteria

1. All TODO sections are correctly implemented
2. Program can handle all three test cases
3. Output results are saved as `output/result_1.png`, `output/result_2.png` and `output/result_3.png`, one per test case
4. Perspective-corrected documents should be horizontal rectangles

## Tips and Suggestions

### Debugging Techniques

1. **Incremental Testing**: Complete one TODO at a time, then test the entire pipeline
2. **Parameter Tuning**: Use different parameter values to observe result changes
3. **Visual Debugging**: Set `show_intermediate=True` to view intermediate results

### Common Issues

1. **Document contour not found**: Check if area threshold is too high, try lowering the threshold
2. **Contour not quadrilateral**: Adjust `cv2.approxPolyDP` epsilon parameter
3. **Perspective transformation failure**: Ensure four vertex coordinates are in correct order

### Advanced Challenges

After completing basic tasks, you can try:
- Automatic Canny threshold adjustment
- Handling multiple documents present simultaneously
- Optimizing processing speed
- Adding image enhancement features

## Submission Requirements

1. Complete `student_document_scanner.py` file
2. Screenshots of running results
3. Brief explanation of chosen parameters and their rationale
4. Summary of problems encountered and solutions during the experiment

## References

- OpenCV Official Documentation: https://docs.opencv.org/
- Canny Algorithm Details: https://en.wikipedia.org/wiki/Canny_edge_detector
- Perspective Transformation Principles: https://en.wikipedia.org/wiki/Homography

---
