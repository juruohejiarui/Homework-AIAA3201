import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# Task 1: Vanilla RNN Cell
# ==========================================
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # For testing, we use fixed weights (no bias)
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x_t, h_prev):
        """
        Calculate the hidden state update for a Vanilla RNN
        Formula: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
        """
        # TODO: Implement the hidden state update for a Vanilla RNN
        # ================= YOUR CODE HERE =================
        
        h_t = torch.tanh(self.W_hh(h_prev) + self.W_xh(x_t)) # Replace this line with your code
        
        # ==================================================
        return h_t

# ==========================================
# Task 2: Scaled Dot-Product Attention
# ==========================================
def scaled_dot_product_attention(q, k, v):
    """
    Calculate the scaled dot-product attention (Scaled Dot-Product Attention)
    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    d_k = q.size(-1)
    
    # TODO: 1. Calculate the alignment scores (Alignment scores)
    # ================= YOUR CODE HERE =================
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # Replace this line with your code
    
    # ==================================================
    
    # TODO: 2. Use softmax to calculate the attention weights (along the last dimension)
    # ================= YOUR CODE HERE =================
    
    attn_weights = F.softmax(scores, dim=-1) # Replace this line with your code
    
    # ==================================================
    
    # TODO: 3. Multiply the attention weights with the value (v) to get the final output
    # ================= YOUR CODE HERE =================
    
    output = torch.matmul(attn_weights, v) # Replace this line with your code
    
    # ==================================================
    
    return output, attn_weights

# ==========================================
# Task 3: Positional Encoding
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # TODO: Calculate the scaling factor (omega_k)
        # ================= YOUR CODE HERE =================
        
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)) # Replace this line with your code
        
        # ==================================================
        
        # TODO: Apply sin to even indices and cos to odd indices
        # ================= YOUR CODE HERE =================
        
        # pe[:, 0::2] = ...
        # pe[:, 1::2] = ...
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        # ==================================================
        
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# ==========================================
# Task 4 & 5: Visualizations
# ==========================================
def plot_positional_encoding():
    print("Saving Positional Encoding plot to ./output ...")
    os.makedirs('./output', exist_ok=True) # Ensure the output directory exists
    
    pe_module = PositionalEncoding(d_model=128, max_len=100)
    pe_matrix = pe_module.pe.squeeze(0).numpy()
    
    plt.figure(figsize=(10, 8))
    # TODO: Use plt.imshow to plot the heatmap of pe_matrix (Heatmap)
    # ================= YOUR CODE HERE =================
    
    plt.imshow(pe_matrix, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1) # Replace this line with your code
    
    # ==================================================
    plt.title("Positional Encoding Heatmap")
    plt.xlabel("Embedding Dimension (d_model)")
    plt.ylabel("Sequence Position (max_len)")
    plt.colorbar()
    
    # Save the image
    save_path = './output/positional_encoding.png'
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close() # Close the figure to free memory

def plot_attention_weights():
    print("Saving Attention Weights plot to ./output ...")
    os.makedirs('./output', exist_ok=True)
    
    seq_len = 10
    d_k = 64
    
    torch.manual_seed(42)
    q = torch.randn(1, seq_len, d_k)
    k = torch.randn(1, seq_len, d_k)
    v = torch.randn(1, seq_len, d_k)
    
    _, attn_weights = scaled_dot_product_attention(q, k, v)
    if attn_weights is None:
        print("Attention weights not implemented yet.")
        return
        
    attn_matrix = attn_weights[0].detach().numpy()
    
    plt.figure(figsize=(6, 5))
    # TODO: Use plt.imshow to plot the heatmap of attn_matrix
    # ================= YOUR CODE HERE =================
    
    plt.imshow(attn_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1) # Replace this line with your code
    
    # ==================================================
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar()
    
    # Save the image
    save_path = './output/attention_weights.png'
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()

# ==========================================
# Test Cases (Run this code block to test your implementation)
# ==========================================
def run_tests():
    print("Running mathematical tests...\n")
    all_passed = True
    
    # Test 1: Vanilla RNN
    try:
        rnn = VanillaRNNCell(input_size=10, hidden_size=20)
        nn.init.ones_(rnn.W_xh.weight)
        nn.init.ones_(rnn.W_hh.weight)
        x_t = torch.ones(2, 10) * 0.5
        h_prev = torch.ones(2, 20) * 0.1
        h_t = rnn(x_t, h_prev)
        expected_ht_value = torch.tanh(torch.tensor(0.5 * 10 + 0.1 * 20))
        assert h_t is not None, "RNN output is None"
        assert torch.allclose(h_t, torch.ones(2, 20) * expected_ht_value), "RNN calculation is incorrect"
        print("✅ Task 1 (Vanilla RNN) Passed!")
    except Exception as e:
        print(f"❌ Task 1 (Vanilla RNN) Failed: {e}")
        all_passed = False

    # Test 2: Scaled Dot-Product Attention
    try:
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        v = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
        out, attn = scaled_dot_product_attention(q, k, v)
        assert out is not None and attn is not None, "Attention output is None"
        assert torch.allclose(attn[0, 0, 0], torch.tensor(0.6698), atol=1e-3), "Attention weights incorrect"
        print("✅ Task 2 (Scaled Dot-Product Attention) Passed!")
    except Exception as e:
        print(f"❌ Task 2 (Scaled Dot-Product Attention) Failed: {e}")
        all_passed = False

    # Test 3: Positional Encoding
    try:
        pe_module = PositionalEncoding(d_model=4, max_len=10)
        dummy_input = torch.zeros(1, 5, 4)
        out = pe_module(dummy_input)
        assert out is not None, "PE output is None"
        assert torch.allclose(out[0, 0, :], torch.tensor([0.0, 1.0, 0.0, 1.0])), "Position 0 encoding incorrect"
        print("✅ Task 3 (Positional Encoding) Passed!")
    except Exception as e:
        print(f"❌ Task 3 (Positional Encoding) Failed: {e}")
        all_passed = False

    # Run Visualizations if maths passed
    print("\n----------------------------------")
    if all_passed:
        print("All mathematical tests passed! Generating visualizations...")
        try:
            plot_positional_encoding()
            plot_attention_weights()
            print("\n🎉 All tests passed and plots are saved in the ./output directory!")
        except Exception as e:
            print(f"Visualization Failed: {e}\nPlease check your matplotlib code.")
    else:
        print("Please fix the mathematical errors before visualizing.")

if __name__ == "__main__":
    run_tests()