import torch
import torch.nn as nn
import random

def run():
    # Reproduce the same randomly selected settings for hand calculation and verification.
    random.seed(42)
    torch.manual_seed(42)
    # Randomly generate reasonable CNN parameters
    N = random.choice([32, 64, 128, 224])
    C_in = random.choice([1, 3])
    C_out = random.choice([16, 32, 64])
    F = random.choice([3, 5, 7])
    stride = random.choice([1, 2])
    padding = random.choice([0, 1, 2, 3])

    print("="*50)
    print("CNN Dimension & Parameter Calculator (Verification Tool)")
    print("="*50)
    print("Given the following input and convolutional layer settings:")
    print(f"- Input Image Size:  {N}x{N}")
    print(f"- Input Channels:    {C_in}")
    print(f"- Output Channels:   {C_out}")
    print(f"- Filter Size (FxF): {F}x{F}")
    print(f"- Stride:            {stride}")
    print(f"- Padding:           {padding}")
    print("-" * 50)
    
    print("STOP! Before looking at the output below, hand-calculate:")
    print("1. The spatial dimensions of the output feature map. Formula: (N + 2P - F) / stride + 1")
    print("2. The total number of parameters in this layer (including bias).")
    print("Once calculated, run this script to verify your answers.")
    print("-" * 50)

    # Create dummy input tensor: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(1, C_in, N, N)

    # ---------------------------------------------------------
    # TODO: Instantiate the 2D Convolutional Layer using nn.Conv2d
    # Pass in the randomly generated parameters defined above.
    # ---------------------------------------------------------
    conv_layer = nn.Conv2d(C_in, C_out, kernel_size=F, stride=stride, padding=padding) 

    if conv_layer is not None:
        # Pass the input through the layer
        output = conv_layer(dummy_input)

        # Calculate total parameters
        total_params = sum(p.numel() for p in conv_layer.parameters())

        expected_size = (N + 2 * padding - F) // stride + 1
        expected_params = C_out * (C_in * F * F + 1)
        assert list(output.shape) == [1, C_out, expected_size, expected_size]
        assert total_params == expected_params
        print(f"Independent formula: floor(({N} + 2*{padding} - {F}) / {stride}) + 1 = {expected_size}")
        print(f"Parameter formula: {C_out} * ({C_in} * {F} * {F} + 1) = {expected_params}")
        print("[VERIFICATION RESULTS]")
        print(f"-> Expected Output Shape:   {output.shape[2]}x{output.shape[3]}x{output.shape[1]} (HxWxC_out)")
        print(f"-> PyTorch Tensor Shape:    {list(output.shape)} (Batch, Channel, Height, Width)")
        print(f"-> Total Layer Parameters:  {total_params}")
    else:
        print("[!] conv_layer is not instantiated. Complete the TODO to see verification results.")
    
    print("="*50)

if __name__ == "__main__":
    run()
