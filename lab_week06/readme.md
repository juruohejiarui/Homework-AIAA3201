# AI Thrust Lab: Building and Inspecting Neural Networks

Welcome to the lab! In this assignment, we will demystify the "black box" of Deep Learning. You will bridge the gap between low-level mathematical logic and high-level framework implementations by completing three specific tasks encompassing Multi-Layer Perceptrons (MLP), Backpropagation, Activation Functions, and Convolutional Neural Networks (CNNs).

## 📌 Submission Requirements
Submit a zip file containing:
1. **Code:** your completed Python scripts (`part1_numpy_backprop.py`, `part2_pytorch_activations.py`, `part3_cnn_dimensions.py`).
2. **Plots:** the `plots/` folder with `activation_comprehensive_analysis.png` from your run.
3. **Report:** a short PDF report, one page is enough, answering two questions:
   * **Part 1:** how many iterations it takes to reach an output close to `[0, 1, 1, 0]`, and what you would try if convergence stays slow.
   * **Part 2:** a short paragraph explaining the difference in convergence between Sigmoid and ReLU.

## 🛠️ Environment Setup
You will need Python 3.8+ and PyTorch installed. 
* **PyTorch Installation:** Visit the [Official PyTorch Get Started Guide](https://pytorch.org/get-started/locally/) and select the command matching your OS and compute platform.
* **Other dependencies:** `pip install numpy matplotlib scikit-learn`
* *Tip:* Setting up deep learning environments can sometimes cause dependency conflicts. Do not hesitate to use Large Language Models (LLMs) to help debug installation errors or interpret Python tracebacks!

---

## Task Details & Instructions

### Part 1: Hand-Coded Backpropagation (XOR Problem)
**File:** `part1_numpy_backprop.py`
A purely linear classifier cannot handle non-linear problems like XOR. In this task, you will implement the forward and backward passes of an MLP using *only* NumPy. We have provided the framework and formatted the output for you.
* **To Do:** Fill in the `TODO` blocks. Implement the Sigmoid function, the forward matrix multiplications, the error derivatives (chain rule), and the weight update step. 
* **Observe:** After running your code for 5 iterations, explicitly observe whether the loss successfully decreases over time.
* **Hint (Forward):** $h = \sigma(XW_1 + b_1)$, $y_{pred} = \sigma(hW_2 + b_2)$
* **Hint (Backward):** Error $E = \frac{1}{2}(y_{pred} - Y)^2$. Start by calculating $\frac{\partial E}{\partial y_{pred}} = (y_{pred} - Y)$, then multiply by the derivative of the Sigmoid function, and backpropagate through the weights.

💡 **Exploration (Optional):**
Try increasing the number of `iterations` in the code (e.g., to 100, 5000, or 10000). Observe the change in convergence speed. 

How many iterations does it take to reach a reasonably accurate output (close to `[0, 1, 1, 0]`)? 

If you find the convergence to be extremely slow or getting "stuck", think about how you might improve it. 
*(Hints to consider: Learning rate tuning? Changing the hidden layer size? Trying a different random seed? The vanishing gradient problem of Sigmoid?)*

### Part 2: Activation Functions Arena
**File:** `part2_pytorch_activations.py`
Are ReLU networks really faster than Sigmoid? You will use PyTorch to build a 3-layer MLP and train it on a non-linear dataset (concentric circles).
* **To Do:** Complete the `__init__` and `forward` methods in the PyTorch `MLP` class. Then, write the standard PyTorch training loop (Forward -> Loss -> Zero Grad -> Backward -> Step).
* The script will automatically train the model twice (once with Sigmoid, once with ReLU), generate a comprehensive plot (including data distribution, loss curves, and decision boundaries), and save it to the `plots/` folder.

### Part 3: CNN Dimension & Parameter Calculator
**File:** `part3_cnn_dimensions.py`
Before training complex CNNs, you must master the fundamental skill of calculating spatial dimensions and parameter counts. 
* **To Do:** Run the script. It will generate randomized settings for an image and a convolutional layer. **Do not write the code yet!** Hand-calculate the output spatial dimensions and the parameter count based on the printed settings.
* **Output Dimension Formula:** $(N + 2P - F) / \text{stride} + 1$
* **Parameter Count Formula:** $C_{out} \times (C_{in} \times F \times F + 1)$
* After calculating by hand, fill in the `TODO` to instantiate `nn.Conv2d` with those parameters. Run the script again to let PyTorch verify if your math was correct. Take a screenshot of the terminal for your report.