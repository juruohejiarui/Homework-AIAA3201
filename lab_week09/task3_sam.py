import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
# We use the hugging face transformers library for easy SAM access without heavy downloads.
from transformers import SamModel, SamProcessor

def run_sam_zeroshot(image, bbox_prompt):
    """
    Run the Segment Anything Model (SAM) using a dynamic bounding box prompt.
    """
    
    print("Loading SAM model (this might take a minute for the first run)...")
    
    # =========================================================================
    # TODO 1: Load the SAM model and its Processor from Hugging Face hub.
    # Hint: Use SamProcessor.from_pretrained and SamModel.from_pretrained 
    # with the exact model ID: "facebook/sam-vit-base"
    # =========================================================================
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base") # Replace
    model = SamModel.from_pretrained("facebook/sam-vit-base") # Replace
    model.eval()
    
    # --- Prompt Preparation and Inference (Provided) ---
    # The processor expects plain Python floats in the format [[[xmin, ymin, xmax, ymax]]].
    bbox_prompt = np.asarray(bbox_prompt, dtype=np.float32).reshape(-1)
    if bbox_prompt.size != 4:
        raise ValueError(f"Expected bbox_prompt with 4 values, got shape {bbox_prompt.shape}")
    input_boxes = [[bbox_prompt.astype(float).tolist()]]
    inputs = processor(image, input_boxes=input_boxes, return_tensors="pt")
    
    # Perform the forward pass to generate masks
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the output masks to match original image sizes
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    
    # =========================================================================
    # TODO 2: Extract the best mask from the post-processed masks array.
    # Hint: 'masks' contains nested arrays. You need the first batch, 
    # first box, and best scoring mask (index 0). Convert it to a boolean numpy array.
    # =========================================================================
    best_mask = masks[0][0][0].cpu().numpy().astype(bool) # Replace
    
    return best_mask

def visualize_task3(image, bbox, mask, output_path):
    """Visualize Box + Real SAM Instance Mask and save."""
    img_copy = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Draw Prompt Bounding Box (provided dynamically by Task 1)
    cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 3) 
    cv2.putText(img_copy, "Dynamic SAM Prompt", (int(x1), max(15, int(y1)-10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    # Apply Mask Overlay (Cyan)
    color = np.array([0, 255, 255]) 
    overlay = img_copy.copy()
    overlay[mask] = color # Use the mask as a boolean indexer
    
    # Blend the overlay and the image
    cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
    
    plt.imsave(output_path, img_copy)
    print(f"[Task 3] Output saved to {output_path}")