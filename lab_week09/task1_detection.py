import torch
import torchvision
from torchvision.transforms import functional as TF
import numpy as np
import cv2
import matplotlib.pyplot as plt

def run_object_detection(image):
    """
    Run real object detection using a pre-trained Faster R-CNN model.
    Only returns the bounding box with the highest confidence score.
    """
    
    # =========================================================================
    # TODO 1: Load the pre-trained Faster R-CNN model with ResNet50-FPN backbone
    # Hint: Use fasterrcnn_resnet50_fpn(pretrained=True)
    # Important: Remember to set the model to evaluation mode using .eval()
    # =========================================================================
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )# Replace
    model.eval()
    
    # --- Data Preprocessing and Inference (Provided) ---
    # Convert image to tensor, add batch dimension, and run forward pass
    img_tensor = TF.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)
    
    pred = predictions[0]
    
    # =========================================================================
    # TODO 2: Find and return ONLY the prediction with the HIGHEST confidence score.
    # Hint 1: Use torch.argmax(pred['scores']) to find the index of the max score.
    # Hint 2: Use slicing [max_idx : max_idx+1] to extract the box, label, and score 
    #         while keeping the array dimensions correct (N, ...).
    # =========================================================================
    if len(pred['scores']) > 0:
        max_idx = int(torch.argmax(pred['scores']).item()) # Replace
        boxes = pred['boxes'][max_idx:max_idx + 1].cpu().numpy() # Replace
        labels = pred['labels'][max_idx:max_idx + 1].cpu().numpy() # Replace
        scores = pred['scores'][max_idx:max_idx + 1].cpu().numpy() # Replace
    else:
        # Return empty arrays if no objects are detected
        boxes, labels, scores = np.array([]), np.array([]), np.array([])
    
    return boxes, labels, scores

def visualize_task1(image, boxes, labels, scores, output_path):
    """Visualize the single best Bounding Box and save the image."""
    img_copy = image.copy()
    
    # Simple COCO mapping for common objects
    coco_names = {1: 'Person', 3: 'Car', 17: 'Cat', 18: 'Dog', 46: 'Bowl', 47: 'Cup'}
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        label_name = coco_names.get(label, f"ID:{label}")
        
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw text with background for better visibility
        text = f"{label_name}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
    plt.imsave(output_path, img_copy)
    print(f"[Task 1] Output saved to {output_path}")