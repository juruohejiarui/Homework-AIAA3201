import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

def run_semantic_segmentation(image):
    """
    Run real semantic segmentation using a pre-trained FCN (Fully Convolutional Network).
    """
    
    # =========================================================================
    # TODO 1: Load the pre-trained FCN-ResNet50 model
    # Hint: Use fcn_resnet50(pretrained=True)
    # Important: Remember to set the model to evaluation mode using .eval()
    # =========================================================================
    model = torchvision.models.segmentation.fcn_resnet50(
        weights=torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
    ) # Replace
    model.eval()
    
    # --- Data Preprocessing and Inference (Provided) ---
    # Define the image transformation pipeline with ImageNet normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transforms, add batch dimension, and run forward pass
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out']
    
    # =========================================================================
    # TODO 2: Obtain the predicted class mask from the model output.
    # Hint: 'output' shape is (1, num_classes, H, W). 
    # Use torch.argmax to collapse the class dimension (dim=0 after squeeze).
    # =========================================================================
    predicted_mask = torch.argmax(output.squeeze(0), dim=0).cpu().numpy() # Replace
    
    return predicted_mask

def visualize_task2(image, mask, output_path):
    """Visualize Semantic Segmentation Mask and save."""
    # Create a simple color palette for Pascal VOC classes
    palette = np.random.randint(0, 255, (21, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0] # Background is black
    
    # Map class IDs to colors
    colored_mask = palette[mask % 21] 
    
    # Overlay mask on original image
    alpha = 0.6
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    plt.imsave(output_path, blended)
    print(f"[Task 2] Output saved to {output_path}")