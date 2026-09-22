---
title: "Lab 3 Report: Document Scanner Implementation"
author: "He Jiarui"
geometry: margin=1in
fontsize: 11pt
---

# 1. Introduction

This lab implements a document-scanning pipeline that locates a quadrilateral document in a cluttered photograph and applies perspective correction. The submitted **baseline** is `student_document_scanner.py` (the four TODOs only). A second file, `student_document_scanner_tryfix.py`, documents extra steps that recover case 2’s outer wood margin and remove case 3’s colored leftover without eating the blue cover. Tryfix outputs are `result_*_tryfix.png`. The TA noted that using the inner engraved border for case 2 is acceptable, so the simpler baseline remains the primary submission; tryfix is included so the repair method can be inspected.

The pipeline is: grayscale conversion, Gaussian blur, Canny edges, largest 4-vertex contour, vertex ordering (top-left, top-right, bottom-right, bottom-left), then `getPerspectiveTransform` + `warpPerspective`.

**Canvas submission.** One zip is uploaded with two code files — `student_document_scanner.py` (baseline) and `student_document_scanner_tryfix.py` (comparison) — and this PDF (`Lab3_Report_HeJiarui.pdf`). The report follows `Submission.md`: labeled results for all three test cases (Section 3), parameter choices with reasoning (Section 2), and a terminal screenshot of a successful run (Section 3.4). Both scripts process all three test cases without errors.

# 2. Implementation Details

## 2.1 Preprocessing — Gaussian blur

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
```

The kernel must be odd. $\sigma = 1.5$ is large enough to suppress desk grain and wood texture before Canny, but small enough that the document outline stays sharp. If $\sigma$ is too small, residual noise becomes false edges. If $\sigma$ is too large, the true boundary is washed out and `approxPolyDP` no longer yields a clean quadrilateral.

## 2.2 Edge detection — Canny

```python
edges = cv2.Canny(image, 50, 150)
```

Canny uses two thresholds and hysteresis: pixels above the high threshold are strong edges; pixels between the two thresholds are kept only if they connect to a strong edge. The assignment suggests setting the low threshold to about one third of the high threshold; $50/150$ follows that ratio. After Canny, the baseline always applies a $5\times5$ morphological close so that a slightly broken outline (especially case 2, where hands and wood grain interrupt the plaque border) still forms one external contour.

## 2.3 Contour filtering

For each external contour:

- skip if `cv2.contourArea(contour)` is below $5\%$ of the image area (desk objects, tree bark);
- approximate with `cv2.approxPolyDP(contour, 0.02 * perimeter, True)`;
- keep the largest contour with exactly four vertices.

$\epsilon = 0.02 \times$ perimeter is the usual starting value: tighter $\epsilon$ leaves too many vertices on rounded corners; looser $\epsilon$ can collapse a real document into a triangle or a non-document shape.

## 2.4 Perspective correction

`order_points` sorts vertices by $x+y$ and $x-y$. The destination rectangle uses the maximum of the two opposite side lengths as width and height. The warp is

```python
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(image, M, (max_width, max_height))
```

## 2.5 Tryfix methods

`student_document_scanner_tryfix.py` keeps the same blur, Canny $50/150$, and 4-gon test, then adds four extras. A uniform 10 px inset was tried first and **rejected**: it pushed case 2 inside the dark frame and shaved case 3’s own blue cover.

1. **Black padding** (`copyMakeBorder`, 20 px) before Canny, then subtract the pad from the vertices, so a page that sits on the image frame (case 3 top/left) still produces an edge.
2. **Morphological close only as a fallback** if raw Canny finds no quadrilateral (needed for case 2’s broken wood outline; always-on close expands case 3 into the colored strip).
3. **Color-connected outer quad (case 2).** After the inner 4-gon is found, take the median HSV of its interior, keep similar pixels, and retain only the connected component that touches that interior (so pavement/tree are ignored). Each inner corner is then walked outward along the ray from the centroid until it leaves this mask. `minAreaRect` was not used: it swallowed the hands and the ground. The expanded quad is accepted only if its area is between $1.05\times$ and $1.45\times$ the inner 4-gon, so case 1/3 do not grow.
4. **Per-side foreign-border trim (case 3).** After warp, each side compares a 1–2 px border strip to a strip about 12 px inward. While they differ (red / near-white / black leftover) and the trim is under 3% of that dimension, the strip is dropped. Trimming stops when the border matches the inward color (case 3: the blue-gray cover). Uniform inset is not used.

These extras are only in the tryfix file; the baseline scanner stays the assignment TODOs.

# 3. Results

All three test images run successfully in both versions. Baseline outputs are `output/result_{1,2,3}.png`. Tryfix outputs are `output/result_{1,2,3}_tryfix.png`.

## 3.1 Case 1 — standard letter on a desk (`asset/case1.png`)

Baseline (`output/result_1.png`, $572\times578$):

![Case 1 baseline](output/result_1.png){width=55%}

Tryfix (`output/result_1_tryfix.png`, $567\times573$):

![Case 1 tryfix](output/result_1_tryfix.png){width=55%}

Both produce an upright rectangle with readable text. Tryfix only nibble-trims a few leftover pixels; the letter is not cut.

## 3.2 Case 2 — wooden plaque, cluttered background (`asset/case2.png`)

Baseline (`output/result_2.png`, $1247\times773$):

![Case 2 baseline](output/result_2.png){width=42%}

Tryfix (`output/result_2_tryfix.png`, $1307\times813$):

![Case 2 tryfix](output/result_2_tryfix.png){width=42%}

The baseline keeps the dark engraved frame and the text inside it; wood from the **physical plaque edge inward to that frame** is discarded. The TA stated that using the inner engraved border as the document boundary is **acceptable**, so the baseline is valid for grading.

Tryfix grows the inner 4-gon through the wood-colored connected component, so the output still shows the dark frame **and** the outer wood (the site URL at the bottom of the plaque becomes visible). Hands and the tree are not included. This is the intended outer crop; it is extra relative to the course’s “inner border is OK” note.

## 3.3 Case 3 — tilted book (`asset/case3.png`)

In the original photograph, the **top and left** of the page meet a **black** background, while the **bottom and right** meet a **short strip of colored** background (gray / red / light tones), not black.

Baseline (`output/result_3.png`, $991\times699$):

![Case 3 baseline](output/result_3.png){width=48%}

The warp is an upright rectangle, but the colored strip on the right and bottom remains. The baseline cannot remove that colored background.

Tryfix (`output/result_3_tryfix.png`, $986\times691$):

![Case 3 tryfix](output/result_3_tryfix.png){width=48%}

Tryfix padding plus **per-side** trim removes the colored leftover on the right and bottom while leaving the book’s own blue-gray cover. An earlier uniform 10 px inset also killed that strip but cut the blue cover; that inset is not in the current tryfix.

## 3.4 Terminal output

Baseline (`python student_document_scanner.py`):

![Baseline terminal](output/terminal_original.png){width=90%}

Tryfix (`python student_document_scanner_tryfix.py`):

![Tryfix terminal](output/terminal_tryfix.png){width=90%}

# 4. Analysis

## 4.1 Case 1

Case 1 is the well-posed setting: a light page on a dark desk, with margin on all sides. Largest-quad selection recovers the letter. Residual dark pixels at the top of the baseline crop come from the photograph’s shading near the paper edge, not from a second object.

## 4.2 Case 2 — inner engraved border vs. physical plaque

The plaque has two nested rectangles: the outer wood edge, and a darker engraved ornamental frame. Canny + `approxPolyDP` prefer the high-contrast inner frame, which is **not** a 4-gon for the outer edge (hands, rounded corners).

- **Baseline:** keeps the dark frame and everything inside it. Wood from the plaque edge to the dark frame is discarded. This matches the TA’s note that the inner engraved border is acceptable.
- **Tryfix:** color-connected growth from the interior, then ray-expand the four corners on that mask. Area is limited to $1.05$–$1.45\times$ the inner quad so the hands/ground cannot win. The warp then includes the outer wood (e.g. `www.woodgeekstore.com` at the bottom) while remaining an upright rectangle.

A first tryfix used a global 10 px inset and `minAreaRect` on a whole-image color mask; that either cropped *inside* the dark frame or wrapped the hands and pavement. Those ideas were dropped.

## 4.3 Case 3 — mixed backgrounds on different sides

Case 3 is asymmetric:

| Side | Adjacent region in the original photo |
|------|----------------------------------------|
| Top, left | Black background |
| Bottom, right | A narrow **colored** background (not black) |

On the black sides, Canny sees a strong page-vs-black edge (or the page is almost flush with the frame). On the colored sides, morphological close in the **baseline** expands the edge into that band, so the largest quadrilateral includes it. After warp, a thin gray/red/light strip remains.

Tryfix (i) pads so the flush top/left still have an edge, (ii) does not close unless necessary, and (iii) after warp trims a side only while its border disagrees with a strip a few pixels inward. The inward reference on the right/bottom is already the blue cover, so trimming stops there instead of eating the cover. The leftover colored background is gone; the blue cover remains.

## 4.4 Challenges and what was kept

1. **Case 2 outline broken by hands and grain.** Baseline always closes the edge map so a 4-gon exists. Tryfix closes only as fallback, then grows by color rather than by dilation.
2. **Case 3 flush / mixed borders.** Uniform inset was too blunt. Per-side color trim after warp is what actually separates “colored leftover” from “blue cover.”
3. **Vertex order.** Using $x+y$ and $x-y$ was sufficient; none of the three warps came out rotated 90°.
4. **What stays in the baseline.** The assignment only asks for the four TODOs. Tryfix is extra and is submitted as a second file, not as a replacement.

# 5. Conclusion

The baseline `student_document_scanner.py` completes the four TODOs and processes all three official test cases without errors. Parameters are $5\times5$ / $\sigma=1.5$ blur, Canny $50/150$, $5\%$ area filter, $\epsilon=0.02$, and always-on $5\times5$ close. Case 2’s inner engraved crop is acceptable per the TA; case 3 still shows a colored strip on the bottom and right.

`student_document_scanner_tryfix.py` addresses those two shortcomings: color-connected corner expansion for case 2’s outer wood, and per-side border trim for case 3’s colored leftover without cutting the blue cover. The Canvas zip contains `student_document_scanner.py`, `student_document_scanner_tryfix.py`, and `Lab3_Report_HeJiarui.pdf`.
