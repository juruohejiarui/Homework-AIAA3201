# Week 01 Lab: Instagram Style Filter Experience

## 📖 Project Introduction

This is the first experiment in the computer vision course. Through implementing Instagram-style filter effects, students experience the instant feedback of computer vision and establish the intuitive understanding that "image processing = numerical operations".

**Lab Objectives**:
- Complete course development environment setup
- Experience instant feedback in computer vision
- Understand the essence of images: pixels and matrices, RGB channels, color spaces

**Student Deliverables**: Generate a montage of multiple filter effects on one photo

## 🛠 Technical Requirements

### Environment Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows / macOS / Linux

### Dependencies
- **OpenCV**: `opencv-python` (image processing)
- **NumPy**: Numerical computing
- **Matplotlib**: Image display

## 📦 Installation Guide

### 1. Check Python Version
```bash
python --version
# Or use on macOS
python3 --version
```

### 2. Install Dependencies
```bash
pip install opencv-python numpy matplotlib
```

**macOS Users**: If you encounter Python version issues, use:
```bash
pip3 install opencv-python numpy matplotlib
```

### 3. Verify Installation
```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)" # python -> python3 for macOS
python -c "import numpy as np; print('NumPy version:', np.__version__)"
```

## 🚀 Usage Instructions

### Quick Start
1. **Download Project**: Ensure all files are in the same directory
2. **Prepare Images**: Place images to process in the `assets/` directory
3. **Run Program**:
   ```bash
   python main.py
   # Or use on macOS
   python3 main.py
   ```
4. **View Results**: Processed images are saved in the `output/` directory

### Example Output
The program generates for each input image:
- **Individual filter images**: 5 filter effect files in `output/{image_name}/`
  - `vintage_warm.jpg` - Vintage warm tone
  - `cool_tone.jpg` - Cool tone filter
  - `high_contrast.jpg` - High contrast
  - `faded.jpg` - Faded effect
  - `grayscale.jpg` - Grayscale
- **Effect montage**: `output/{image_name}/montage.jpg` - 2x3 collage with original and 5 filter effects

## 📁 Code Structure

```
lab_week01/
├── assets/           # Input images directory
│   ├── human.jpg     # Sample portrait photo
│   ├── scenery.jpg   # Sample landscape photo
│   └── object.jpg    # Sample object photo
├── filters.py        # Filter implementation functions
├── main.py          # Main program
├── output/          # Output results directory (generated after running)
│   ├── human/       # Results for human.jpg
│   │   ├── vintage_warm.jpg
│   │   ├── cool_tone.jpg
│   │   ├── high_contrast.jpg
│   │   ├── faded.jpg
│   │   ├── grayscale.jpg
│   │   └── montage.jpg
│   └── ...          # Similar structure for other images
└── README.md        # This documentation
```

### Core Files Description

#### `filters.py`
Contains implementation of 5 Instagram-style filters:
- `apply_vintage_warm()` - Vintage warm tone
- `apply_cool_tone()` - Cool tone filter
- `apply_high_contrast()` - High contrast
- `apply_faded()` - Faded effect
- `apply_grayscale()` - Grayscale conversion

#### `main.py`
Main program logic:
- Reads images from `assets/` directory
- Applies all filter effects
- Generates effect montage
- Saves results to `output/` directory

## 🎨 Filter Effects Description

| No. | Filter Name | Effect Description | Implementation Principle |
|-----|------------|-------------------|------------------------|
| 1 | Vintage Warm | Adds yellow-orange tones, simulates film texture | RGB channel weighted adjustment |
| 2 | Cool Tone | Adds blue tones, creates cool atmosphere | Enhances B channel, reduces R/G channels |
| 3 | High Contrast | Enhances light-dark contrast for impact | Stretches histogram, increases dynamic range |
| 4 | Faded | Reduces contrast and saturation for nostalgic feel | Compresses dynamic range, reduces saturation |
| 5 | Grayscale | Converts to black and white, classic effect | RGB → Grayscale conversion |

## ❓ Common Issues and Solutions

### Issue 1: OpenCV Installation Failed
**Symptoms**: `pip install opencv-python` reports error or timeout

**Solutions**:
- Use domestic mirror source:
  ```bash
  pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- If still fails, try installing `opencv-python-headless`:
  ```bash
  pip install opencv-python-headless
  ```

### Issue 2: Image Path Not Found
**Symptoms**: Code reports "File not found"

**Solutions**:
- Check if image files are in the `assets/` directory
- Use absolute paths or confirm relative paths are correct
- Note: Windows uses backslashes `\` (recommended to use `/` uniformly)

### Issue 3: Generated Images Cannot Display
**Symptoms**: Code runs successfully but window flashes

**Solutions**:
- Program saves results to files by default, open with system image viewer
- For real-time display, add display logic to the code

### Issue 4: macOS Python Version Confusion
**Symptoms**: System has Python 2.7, installed Python 3.x not usable

**Solutions**:
- Use `python3` and `pip3` commands instead of `python` and `pip`
- Or configure alias: `alias python=python3`

### Issue 5: Permission Error
**Symptoms**: pip installation prompts permission error

**Solutions**:
```bash
# Use user-level installation
pip install --user opencv-python numpy matplotlib
```

## 📚 Reference Resources

### Official Documentation
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy Documentation](https://numpy.org/doc/)

### Extended Reading
- Color Space Conversion: Differences and applications of RGB, HSV, LAB
- Histogram Equalization: Classic method to enhance image contrast

### Advanced Challenges (Optional)
After completing basic requirements, students can try:
1. Modify filter parameters to create unique styles
2. Add new filter effects (such as sharpening, edge detection)
3. Implement real-time filter preview (using webcam)

---

## 💡 Learning Tips

- **Understand Image Essence**: Images in computers are digital matrices, each pixel has RGB channel values
- **Lab Focus**: Not learning specific filter algorithms, but establishing "image processing = numerical operations" concept
- **Encourage Exploration**: Try modifying filter parameters and observe different visual effects

Ask TA anytime if you have questions! 🚀
