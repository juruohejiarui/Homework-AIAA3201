---
title: "Lab 3 Report: Document Scanner Implementation"
author: "He Jiarui"
geometry: margin=1in
fontsize: 11pt
---

# 1. Introduction

This lab implements a document-scanning pipeline that locates a quadrilateral document in a cluttered photograph and applies perspective correction. The submitted baseline is `student_document_scanner.py`. An experimental variant is kept as `student_document_scanner_tryfix.py` (the version previously stored in `lab_week03_hand`), with outputs named `result_*_tryfix.png`. The tryfix pipeline is reported for comparison only; it is **not** treated as a successful fix of the baseline limitations, for the reasons in Section 4.

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

## 2.5 What the tryfix version adds

`student_document_scanner_tryfix.py` keeps the same blur, Canny thresholds, and contour test, then adds three extras aimed at case 3:

1. **Black padding** (`copyMakeBorder`, 20 px) before Canny, then subtract the pad from the vertices, so a page that sits on the image frame still produces an edge.
2. **Morphological close only as a fallback** if no quadrilateral is found on raw Canny (so case 3 is not dilated outward by default).
3. **A 10 px inset** of the ordered quadrilateral before warping, to drop a thin leftover strip on exposed sides.

These changes are discussed as a failed / incomplete repair in Section 4, not as the submitted solution.

# 3. Results

All three test images run successfully in both versions. Baseline outputs are `output/result_{1,2,3}.png`. Tryfix outputs are `output/result_{1,2,3}_tryfix.png`.

## 3.1 Case 1 — standard letter on a desk (`asset/case1.png`)

Baseline (`output/result_1.png`, $572\times578$):

![Case 1 baseline](output/result_1.png){width=55%}

Tryfix (`output/result_1_tryfix.png`, $551\times555$):

![Case 1 tryfix](output/result_1_tryfix.png){width=55%}

Both produce an upright rectangle with readable text. The tryfix inset crops a few extra pixels of the paper margin.

## 3.2 Case 2 — wooden plaque, cluttered background (`asset/case2.png`)

Baseline (`output/result_2.png`, $1247\times773$):

![Case 2 baseline](output/result_2.png){width=42%}

Tryfix (`output/result_2_tryfix.png`, $1225\times753$):

![Case 2 tryfix](output/result_2_tryfix.png){width=42%}

The baseline keeps the dark engraved frame and the text inside it; the wood between the physical plaque edge and that dark frame is cropped away. The tryfix result is tighter still: it mainly keeps content **inside** the dark frame, so the frame itself is partly lost. This is not treated as a repair of the outer-boundary problem (Section 4.2). The course clarification that using the inner engraved border as the document boundary is acceptable for case 2 applies to the **baseline**.

## 3.3 Case 3 — tilted book (`asset/case3.png`)

In the original photograph, the **top and left** of the page meet a **black** background, while the **bottom and right** meet a **short strip of colored** background (gray / red / light tones), not black.

Baseline (`output/result_3.png`, $991\times699$):

![Case 3 baseline](output/result_3.png){width=48%}

The warp is an upright rectangle, but the colored strip on the right and bottom remains. The baseline cannot remove that colored background.

Tryfix (`output/result_3_tryfix.png`, $970\times677$):

![Case 3 tryfix](output/result_3_tryfix.png){width=48%}

The tryfix inset/padding can suppress the guessed colored background, but it also cuts into the book’s own blue-gray cover. That is a trade-off, not a clean crop to the true page edge.

## 3.4 Terminal output

Baseline (`python student_document_scanner.py`):

![Baseline terminal](output/terminal_original.png){width=90%}

Tryfix (`python student_document_scanner_tryfix.py`):

![Tryfix terminal](output/terminal_tryfix.png){width=90%}

# 4. Analysis

## 4.1 Case 1

Case 1 is the well-posed setting: a light page on a dark desk, with margin on all sides. Largest-quad selection recovers the letter. Residual dark pixels at the top of the baseline crop come from the photograph’s shading near the paper edge, not from a second object.

## 4.2 Case 2 — inner engraved border vs. physical plaque

The plaque has two nested rectangles: the outer wood edge, and a darker engraved ornamental frame. Canny + `approxPolyDP` prefer the high-contrast inner frame. Consequences:

- **Baseline:** the output keeps the dark frame and everything inside it. Material from the **physical paper/plaque edge inward to the dark frame** is discarded.
- **Tryfix:** because close is delayed and the quad is inset by 10 px, the crop moves further inward and mainly retains content **inside** the dark frame.

The tryfix therefore does **not** restore the missing outer margin. It does not fix the case 2 boundary error; it can make the inner crop stricter. For that reason the tryfix is not regarded as a solution.

The instructor noted that **for case 2, using the inner engraved border as the document boundary is acceptable**. Under that criterion the baseline output is valid. The remaining limitation is only that a scanner that aimed at the physical outer edge would still fail; the tryfix does not achieve that goal either.

## 4.3 Case 3 — mixed backgrounds on different sides

Case 3 is asymmetric:

| Side | Adjacent region in the original photo |
|------|----------------------------------------|
| Top, left | Black background |
| Bottom, right | A narrow **colored** background (not black) |

On the black sides, Canny sees a strong page-vs-black edge (or the page sits so close to the image frame that the edge is weak). On the colored sides, morphological close in the baseline **expands** the edge into that colored band, so the largest quadrilateral includes it. After warp, a thin gray/red/light strip remains; the baseline cannot delete it.

The tryfix (padding, close-as-fallback, 10 px inset) can remove that guessed colored background, but the same inset also **cuts part of the paper’s own blue cover**. The colored strip is gone at the cost of eating the document. That is why the tryfix is reported as an experiment, not as the submitted fix: it trades one boundary error for another.

No single global Canny / close / inset setting was found that (i) keeps case 2’s inner frame (or better, the outer plaque), (ii) drops case 3’s colored strip, and (iii) leaves case 3’s blue cover intact.

## 4.4 Challenges and what was kept

1. **Case 2 outline broken by hands and grain.** Without morphological close, the baseline often finds no 4-gon. Always-on close was kept so all three cases run.
2. **Case 3 flush / mixed borders.** Padding and inset were tried (`*_tryfix`) and then rejected as the main code, because they did not honestly fix case 2 and they over-crop case 3’s blue margin.
3. **Vertex order.** Using $x+y$ and $x-y$ was sufficient; none of the three warps came out rotated 90°.

# 5. Conclusion

The baseline `student_document_scanner.py` completes the four TODOs and processes all three official test cases without errors, producing horizontal rectangles. Parameters are $5\times5$ / $\sigma=1.5$ blur, Canny $50/150$, $5\%$ area filter, $\epsilon=0.02$, and always-on $5\times5$ close. The Canvas zip contains `student_document_scanner.py`, `student_document_scanner_tryfix.py`, and `Lab3_Report_HeJiarui.pdf`.

Known shortcomings: case 2 locks onto the inner engraved frame (acceptable per the instructor, but the outer wood is lost); case 3 keeps a colored background strip on the bottom and right. The tryfix variant can hide that strip but over-crops case 2 into the interior of the dark frame and shaves case 3’s own blue cover, so it is not considered a real repair. The submitted scanner is the baseline; the tryfix code and `result_*_tryfix.png` figures are included only as documentation of that attempt.
