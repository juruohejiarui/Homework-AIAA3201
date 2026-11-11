# Computer Vision Lab: Real Inference with Pre-trained Models & SAM

Welcome to the Detection & Segmentation lab! In this updated assignment, we will move beyond theoretical architectures and get hands-on experience with **real-world inference pipelines**. You will learn how to load state-of-the-art pre-trained models, properly preprocess image data, run forward passes, and extract meaningful predictions for Object Detection, Semantic Segmentation, and Zero-Shot Instance Segmentation using Foundation Models.

## 🎯 Learning Objectives
By completing this lab, you will:
1. Understand how to utilize `torchvision` to load pre-trained Computer Vision models (Faster R-CNN, FCN-ResNet50).
2. Learn the critical importance of image preprocessing (e.g., ImageNet normalization, Tensor conversion) before feeding data into neural networks.
3. Extract and parse model outputs (bounding boxes, confidence scores, class logits) into visualizable formats.
4. Gain hands-on experience with the Segment Anything Model (SAM) using the Hugging Face `transformers` library to perform prompt-based instance segmentation.

---

## 📁 Project Structure

Please do not modify the file names or the `main.py` / `utils.py` execution logic. You only need to write code in the files marked with `[Action Required]`.

```text
├── input/                      # Place the input images used by `main.py` here
├── output/                     # Directory where your visualized results will be saved
├── utils.py                    # Helper functions (image loading, directory setup)
├── main.py                     # Main execution script to run all tasks
├── task1_detection.py          # [Action Required] Faster R-CNN inference module
├── task2_segmentation.py       # [Action Required] FCN-ResNet50 inference module
└── task3_sam.py                # [Action Required] SAM Zero-shot inference module
```

---

## 🛠️ Environment Setup

Ensure you have a Python environment set up with the following dependencies. You can install them via pip. **Note the addition of the `transformers` library for Task 3.**

```bash
pip install torch torchvision transformers opencv-python numpy matplotlib
```

---

## 📝 Lab Instructions

Search for the **`TODO:`** comments in the respective Python files to find exactly where you need to implement your code. 

### Task 1: Object Detection Inference (Faster R-CNN)
**File:** `task1_detection.py`

* **Your task:** Load a pre-trained `fasterrcnn_resnet50_fpn` model from `torchvision`. 
* **Data Processing:** Convert the standard numpy image array into a PyTorch Tensor and add a batch dimension.
* **Output Parsing:** The model will output many candidate predictions. You need to find the prediction with the **highest confidence score** and return only that single bounding box, label, and score while keeping the array dimensions correct.

### Task 2: Semantic Segmentation Inference (FCN-ResNet50)
**File:** `task2_segmentation.py`

* **Your task:** Load a pre-trained `fcn_resnet50` model.
* **Data Processing:** Segmentation models are highly sensitive to input normalization. You must define a `transforms.Compose` pipeline that converts the image to a tensor AND normalizes it using standard ImageNet statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
* **Output Parsing:** The model outputs class logits for every pixel. You need to use `torch.argmax` to collapse the channel dimension and obtain the final predicted class mask.

### Task 3: Zero-Shot Transfer with SAM (Foundation Models)
**File:** `task3_sam.py`

* **Your task:** Load the lightweight `facebook/sam-vit-base` model and its corresponding Processor using Hugging Face's `transformers` library.
* **Prompting:** Format the given bounding box into the nested list structure required by the SAM processor (`[[[xmin, ymin, xmax, ymax]]]`).
* **Inference:** Pass the image and the bounding box prompt through the model, and use the processor's post-processing function to extract the precise pixel mask of the target object.
* **Output Parsing:** From the post-processed SAM output, extract the first batch, first box, and best mask (index `0`), then convert it to a boolean numpy mask for visualization.

---

## 🚀 How to Run and Test Your Code

1. **Prepare your images**: Place three images at the paths currently used by `main.py`: `input/single_object_detection.png`, `input/semantic_segmentation.png`, and `input/SAM_simulation.png`.
2. **Bounding Box Flow**: You do **not** need to manually edit Bounding Box coordinates in `main.py` for the current project code. In Task 3, the pipeline first runs object detection on the Task 3 image, then passes the highest-confidence bounding box to SAM as the prompt.
3. **Run the script**:

```bash
python main.py
```

*Note: The first time you run this script, it will download the pre-trained weights for ResNet and SAM (around 500MB total). Please ensure you have a stable internet connection.*

**Expected Behavior:**
If your implementations are correct, the script will process your images through the real neural networks and generate three visualization images in the `output/` folder:
- `task1_detection_result.jpg`: Shows accurate bounding boxes around detected objects.
- `task2_segmentation_result.jpg`: Shows a color-coded semantic mask overlay mapping different categories (cars, people, background).
- `task3_sam_real_result.jpg`: Shows the orange prompt box generated by the pipeline and a precise cyan instance mask hugging the object's edges.

---

## 📬 Submission Guidelines
Please upload your final output images in JPG or PNG format to Canvas.