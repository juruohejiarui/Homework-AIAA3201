# Lab 3 Submission Guidelines

## Overview

This document outlines the submission requirements for Week 03 Lab Assignment: Document Scanner Implementation. All submissions must be made through Canvas before the next class session.

## Submission Files

### 1. Code File
- **File Name**: `student_document_scanner.py`
- **Requirements**:
  - Complete all TODO sections in the code
  - Ensure the code runs without errors
  - Follow the implementation guidelines provided in LAB_ASSIGNMENT.md
  - Include proper comments explaining your implementation choices

### 2. Lab Report
- **File Name**: `Lab3_Report_YourName.pdf` (replace YourName with your actual name)
- **Requirements**: The report must include the following sections:

#### 2.1 Output Results
- Screenshots of all processed images from the `output/` directory
- Include results for all three test cases:
  - `asset/case1.png` → `output/result_1.png`
  - `asset/case2.png` → `output/result_2.png`
  - `asset/case3.png` → `output/result_3.png`
- Clearly label each image with the corresponding test case

#### 2.2 Analysis and Explanation
- **Parameter Choices**: Explain the parameter values you chose for each TODO section
  - Gaussian blur kernel size and sigma values
  - Canny edge detection thresholds
  - Contour filtering criteria
- **Implementation Decisions**: Describe your approach to solving each TODO section
- **Challenges Encountered**: Discuss any difficulties you faced and how you resolved them
- **Results Analysis**: Comment on the quality of the output images and any limitations

#### 2.3 Terminal Screenshots
- Screenshot showing successful execution of the program
- Include the complete output from running `python student_document_scanner.py`
- Demonstrate that all test cases process successfully

## Submission Format

### File Structure
When submitting, organize your files as follows:
```
Lab3_Submission/
├── student_document_scanner.py
└── Lab3_Report_YourName.pdf
```

### Canvas Submission
1. Compress both files into a single ZIP archive (required, the Canvas assignment only accepts zip)
2. Upload to Canvas under the Week-03 assignment
3. Ensure file names follow the exact format specified above

## Important Notes

- **Deadline**: Submit before the next class session
- **File Naming**: Strictly follow the naming conventions specified
- **Code Functionality**: Your code must successfully process all three test images
- **Original Work**: All code and analysis must be your own work
- **Late Submissions**: Will be subject to late penalties as per course policy

## Example Report Structure

```
Lab3_Report_JohnDoe.pdf

1. Introduction
   - Brief overview of the assignment

2. Implementation Details
   - Parameter choices and reasoning
   - Approach to each TODO section

3. Results
   - Screenshots of all output images
   - Terminal execution screenshots

4. Analysis
   - Success of each test case
   - Challenges and solutions
   - Limitations and potential improvements

5. Conclusion
   - Summary of learning outcomes
```
