# AIAA3201 Lab: Recurrent Neural Networks and Transformers

## Lab Overview
This lab section aims to help you translate the mathematical formulas of sequence models and attention mechanisms from the lecture slides into actual PyTorch code. By manually implementing the core components of RNNs and Transformers, you will gain a deeper, code-level understanding of how these powerful architectures process sequential data.

## Task Description

### Part 1: Vanilla RNN Cell
- Implement the single-step hidden state update rule for a Vanilla RNN.
- **Formula**: $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$
- You need to properly apply matrix multiplications and the non-linear activation function.

### Part 2: Scaled Dot-Product Attention
- Implement the core mechanism of the Transformer architecture: Scaled Dot-Product Attention.
- **Formulas**: 
  - Alignment scores: $e = \frac{Q K^T}{\sqrt{d_k}}$
  - Attention weights: $a = \text{softmax}(e)$
  - Context vector: $y = a V$
- Pay attention to the scaling factor and the dimension along which the softmax is applied.

### Part 3: Positional Encoding
- Implement the sinusoidal positional encoding to inject sequence order information into the Transformer.
- **Formula**: $\omega_k = \frac{1}{10000^{2k/d_{model}}}$
- Apply $\sin$ to even indices and $\cos$ to odd indices of the encoding matrix.

### Part 4: Visualizations
- After passing the mathematical tests, you will implement matplotlib code to visualize:
  1. **Positional Encoding Heatmap**: Observe the frequency variations across different dimensions.
  2. **Attention Weights Heatmap**: Understand how queries and keys interact to form a probability distribution.
- The generated plots will be automatically saved to the `/output` directory.

## Getting Started
1. Open `student_lab.py` and locate all the `TODO` blocks.
2. Complete the code implementations for Parts 1 to 4.
3. Run the verification script in your terminal:
   ```bash
   python3 student_lab.py
   ```
4. Ensure all mathematical test cases show the ✅ mark, and verify that `positional_encoding.png` and `attention_weights.png` are successfully generated in the `./output/` directory.


## Submission Guideline

Submit a zip file containing:
1. **Code**: your completed `student_lab.py`.
2. **Plots**: the `output/` folder with `positional_encoding.png` and `attention_weights.png` from your run.
3. **Report**: a short PDF report, one page is enough, answering two questions:
   * Why is the scaling factor $\sqrt{d_k}$ necessary in scaled dot-product attention?
   * What pattern do you observe in the positional encoding heatmap?

---
Note: Please ensure your Python environment has `torch` and `matplotlib` installed before starting the lab.