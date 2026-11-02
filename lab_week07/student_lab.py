import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# Part 1: Activation Functions
# ==========================================
class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        """
        Forward pass for Sigmoid.
        Formula: sigma(x) = 1 / (1 + e^(-x))
        """
        out = None
        # --- START CODE HERE ---
        out = 1.0 / (1.0 + np.exp(-x))
        # --- END CODE HERE ---
        return out

    @staticmethod
    def sigmoid_backward(dout, x):
        """
        Backward pass for Sigmoid.
        Derivative: sigma(x) * (1 - sigma(x))
        """
        dx = None
        # --- START CODE HERE ---
        sigmoid = ActivationFunctions.sigmoid(x)
        dx = dout * sigmoid * (1.0 - sigmoid)
        # --- END CODE HERE ---
        return dx

    @staticmethod
    def relu(x):
        """
        Forward pass for ReLU.
        Formula: max(0, x)
        """
        out = None
        # --- START CODE HERE ---
        out = np.maximum(0, x)
        # --- END CODE HERE ---
        return out

# ==========================================
# Part 2: Advanced Optimizers
# ==========================================
class Optimizers:
    @staticmethod
    def sgd_momentum(w, dw, v, lr=1e-3, rho=0.9):
        """
        SGD with Momentum.
        v = rho * v + dw
        w = w - lr * v
        """
        # --- START CODE HERE ---
        v = rho * v + dw
        w = w - lr * v
        # --- END CODE HERE ---
        return w, v

    @staticmethod
    def adam(w, dw, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7):
        """
        Adam Optimizer with bias correction.
        1. Update biased first moment estimate 'm'
        2. Update biased second raw moment estimate 'v'
        3. Compute bias-corrected first moment estimate 'm_unbias'
        4. Compute bias-corrected second raw moment estimate 'v_unbias'
        5. Update weights 'w'
        """
        # --- START CODE HERE ---
        m = beta1 * m + (1.0 - beta1) * dw
        v = beta2 * v + (1.0 - beta2) * np.square(dw)
        m_unbias = m / (1.0 - beta1 ** t)
        v_unbias = v / (1.0 - beta2 ** t)
        w = w - lr * m_unbias / (np.sqrt(v_unbias) + eps)
        # --- END CODE HERE ---
        return w, m, v

# ==========================================
# Part 3: Regularization (Inverted Dropout)
# ==========================================
class Dropout:
    @staticmethod
    def dropout_forward(x, p=0.5, mode='train'):
        """
        Inverted Dropout forward pass.
        'p' is the probability of KEEPING a neuron active.
        """
        mask = None
        out = None
        
        if mode == 'train':
            # --- START CODE HERE ---
            # 1. Create a binary mask (1 with probability p, 0 with probability 1-p)
            # 2. Scale the mask by 1/p (Inverted Dropout)
            # 3. Apply the mask to input x
            mask = (np.random.rand(*x.shape) < p) / p
            out = x * mask
            # --- END CODE HERE ---
        elif mode == 'test':
            # --- START CODE HERE ---
            # In inverted dropout, test time is just identity
            out = x
            # --- END CODE HERE ---
            
        return out, mask

# ==========================================
# Part 4: Batch Normalization (Simplified 1D)
# ==========================================
class BatchNorm:
    @staticmethod
    def batchnorm_forward(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.9, mode='train'):
        """
        Batch Normalization forward pass.
        """
        out = None
        if mode == 'train':
            # --- START CODE HERE ---
            # 1. Calculate batch mean and variance
            # 2. Normalize: x_norm = (x - mean) / sqrt(var + eps)
            # 3. Apply scale (gamma) and shift (beta)
            # 4. Update running_mean and running_var using momentum
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_norm = (x - mean) / np.sqrt(var + eps)
            out = gamma * x_norm + beta
            running_mean = momentum * running_mean + (1.0 - momentum) * mean
            running_var = momentum * running_var + (1.0 - momentum) * var
            # --- END CODE HERE ---
            
        elif mode == 'test':
            # --- START CODE HERE ---
            # Normalize using running_mean and running_var
            x_norm = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * x_norm + beta
            # --- END CODE HERE ---
            
        return out, running_mean, running_var

# ==========================================
# Part 5: Transfer Learning (PyTorch)
# ==========================================
def setup_transfer_learning(num_classes=10):
    """
    Load a pre-trained ResNet18, freeze its convolutional layers, 
    and replace the final fully connected layer.
    """
    model = None
    # --- START CODE HERE ---
    # 1. Load pre-trained ResNet18 (weights=models.ResNet18_Weights.DEFAULT)
    # 2. Freeze all parameters by setting requires_grad = False
    # 3. Replace model.fc with a new nn.Linear layer for num_classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # --- END CODE HERE ---
    
    return model
