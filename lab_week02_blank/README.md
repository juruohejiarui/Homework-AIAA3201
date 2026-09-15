# Week 02 Lab: Image Filters and Hybrid Images

## Learning Objectives

- Understand what a 2D linear filter really does, by writing the convolution yourself
- Learn when to use a Gaussian filter and when to use a median filter for denoising
- Understand low frequency and high frequency components, and build a hybrid image

## Environment

Same as Lab 1. Python 3.8+, with OpenCV and NumPy.

```bash
pip install opencv-python numpy
```

## Project Structure

```
lab_week02/
├── assets/       # Input images (image1, image2, image3a, image3b, scenery)
├── filter.py     # Task 1: manual convolution (TODO inside)
├── denoise.py    # Task 2: denoising battle (TODO inside)
├── hybrid.py     # Task 3: hybrid images (TODO inside)
├── main.py       # Runner. Do not modify unless you try your own images
├── output/       # Results appear here after running
└── README.md
```

Find the sections marked `# TODO: Students ...` in the three files and complete them. Hints are provided in the comments.

## Tasks

### Task 1: Implement my_imfilter (filter.py)

Write a 2D convolution from scratch. **Do not use cv2.filter2D or any library convolution routine in this task.** The point is to see what happens under the hood.

Two steps, both marked TODO in the file.

1. **Padding.** Use `np.pad` with `mode='reflect'` so the kernel fits at the image borders. For a color image, pad the height and width axes only, not the channel axis.
2. **Convolution loop.** Slide the kernel over every pixel. At each position, take the region of the padded image, multiply element-wise with the kernel, sum, and write the result to the output. For color images, process each channel independently.

`main.py` sanity checks your implementation on `assets/image1.jpg` with an identity kernel (output should equal input) and a 3x3 box kernel (output should be slightly blurred).

**Discussion.** Why reflect padding instead of zero padding? Look at the border of `part1_box_blur.jpg` and imagine what zero padding would do there.

### Task 2: Denoising Battle (denoise.py)

Fill in the two API calls in `denoise_image()`. This task is meant to use OpenCV functions.

- `'gaussian'` → `cv2.GaussianBlur` (set sigmaX to 0 so it is derived from the kernel size)
- `'median'` → `cv2.medianBlur`

`main.py` adds Gaussian noise (sigma 25) and salt & pepper noise (5%) to `assets/image2.jpg`, denoises each with both filters, and saves six comparison images.

**What you should observe.**

1. Gaussian noise. The Gaussian filter smooths it well. The median filter also works but can look painted or cartoonish.
2. Salt & pepper noise. The Gaussian filter fails (it only smears the black and white dots). The median filter removes them cleanly.

**Discussion.** Why does a median (nonlinear) filter succeed on impulse noise where a linear filter fails?

### Task 3: Hybrid Images (hybrid.py)

Complete `create_hybrid_image()` at the three TODO marks.

1. `low_frequencies` = Gaussian blur of image1. The kernel size `k_size` is already computed for you, and the cutoff frequency is the sigma.
2. `high_frequencies` = image2 minus its own blurred (low-pass) version.
3. `hybrid_image` = low frequencies of image1 + high frequencies of image2.

`main.py` combines `assets/image3a.jpg` (low frequency) with `assets/image3b.jpg` (high frequency) into `output/part3_hybrid.jpg`. View it from far away or squint at it, then view it up close.

**Optional challenge.** Make your own hybrid image from your daily life (for example a friend's face blended with a cat, or two fruits). The secret to a convincing illusion is **alignment** of key features such as eyes, nose and mouth. Put your pair in `assets/`, point `main.py` at them, and keep your extra result in `output/`.

## Expected Output

After `python main.py` (or `python3 main.py` on macOS), the `output/` folder should contain nine images.

| File | From |
|---|---|
| part1_identity.jpg | Task 1, identity kernel |
| part1_box_blur.jpg | Task 1, box blur |
| task2_A_noisy_gaussian.jpg | Task 2, noisy input |
| task2_A_restore_by_gaussian.jpg | Task 2 |
| task2_A_restore_by_median.jpg | Task 2 |
| task2_B_noisy_sp.jpg | Task 2, noisy input |
| task2_B_restore_by_gaussian.jpg | Task 2 |
| task2_B_restore_by_median.jpg | Task 2 |
| part3_hybrid.jpg | Task 3 |

## Submission

Zip your **whole `lab_week02` folder** (your completed code plus the `output` folder) and upload it to the Lab 2 assignment on Canvas. Keep the folder name as `lab_week02`, so the paths inside your zip look like `lab_week02/filter.py` and `lab_week02/output/part1_identity.jpg`.

Grading checks that the three tasks are implemented correctly, that Task 1 does not call library convolution, and that the output folder contains the nine images with correct effects.

Work individually. If you get stuck, raise your hand during the lab session.
