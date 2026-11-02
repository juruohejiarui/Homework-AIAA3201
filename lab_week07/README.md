# AIAA3201 Lab Week 07: CNN Optimization, Regularization, and Advanced Architectures

## Lab Overview
This lab section aims to help you translate mathematical formulas and algorithm theories from slides into actual code. By manually implementing common "black-box" operations, you will gain a deeper understanding of how deep learning models are optimized and regularized.

## Task Description

### Part 1: Activation Functions
- **Sigmoid**: Implement forward pass ($\sigma(x) = 1 / (1 + e^{-x})$) and its backward derivative ($\sigma(x)(1-\sigma(x))$).
- **ReLU**: Implement forward pass ($max(0, x)$).

### Part 2: Advanced Optimizers
- **SGD + Momentum**: Implement the velocity update rule.
- **Adam**: Implement first and second moment updates with **Bias Correction**.

### Part 3: Regularization - Inverted Dropout
- Implement **Inverted Dropout** forward pass:
  - In `train` mode, apply a binary mask and scale by $1/p$.
  - In `test` mode, return the input as-is (Identity).

### Part 4: Batch Normalization (1D)
- Implement simplified Batch Normalization forward pass:
  - In `train` mode, calculate batch statistics and update `running_mean`/`running_var`.
  - In `test` mode, normalize using the accumulated running statistics.

### Part 5: Transfer Learning (PyTorch)
- Use `torchvision` to load a pre-trained ResNet18.
- Freeze the feature extraction layers (`requires_grad = False`).
- Replace the final classification head (`model.fc`) for a new number of classes.

## Getting Started
1. Complete the `TODO` sections in `student_lab.py`.
2. Run the verification script:
   ```bash
   python3 main.py
   ```
3. Ensure all sections show the ✅ mark.

## Submission Guideline
Submit a zip file containing:
1. **Code**: your completed `student_lab.py`.
2. **Report**: a short PDF report, one page is enough, answering two questions:
   * Why is bias correction needed in Adam?
   * Why does inverted dropout return the input unchanged at test time, and why scale the mask by 1/p during training?

---
**Deadline**: Please refer to the course website for the specific deadline.
