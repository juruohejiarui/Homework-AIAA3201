# Lab 06 submission

Completed in conda environment `aiaa3201`, as confirmed after the original `aiaa3102` environment was not found. All ten original TODO comments remain for review. The unchanged original lab package is archived locally as `../lab_week06_hand/` and excluded from this submission.

## Run

```sh
conda activate aiaa3201
python main.py
```

Run from the extracted ZIP directory. The default Part 1 run prints five iterations. Part 2 trains both models for 1,000 epochs and writes `plots/activation_comprehensive_analysis.png`. Part 3 uses seed 42 so the hand-calculated settings are reproduced. `main.py --part 1`, `--part 2`, and `--part 3` run each part separately.

The tested dependency versions are in `requirements.txt`; the Python interpreter and version are recorded in `logs/environment.txt`. On an existing compatible Python environment, dependencies can be installed with `python -m pip install -r requirements.txt`. These exact pins describe the tested Python 3.13/macOS setup, rather than promising compatibility with every Python 3.8+ release.

For headless runs or a read-only home directory, set `MPLBACKEND=Agg` and set `MPLCONFIGDIR` to a writable cache directory. CPU thread counts were set to one during verification (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`).

Reproduce the Part 1 convergence measurement:

```sh
python part1_numpy_backprop.py --iterations 200000 --quiet --stop-at-tolerance
```

The reported update count is the first **post-update** state with all four absolute errors <= 0.1: 8,304 updates, predictions `[0.090604, 0.900009, 0.900158, 0.080664]`. The default five-row table uses **pre-update** predictions, as in the original template. The loss is half the mean squared error; gradients include averaging over the four samples. The conditional width-4 experiment was unnecessary because the baseline met the threshold.

Part 2 uses identical initial linear parameters for Sigmoid and ReLU (seed reset to 42 for each network), full-batch SGD with learning rate 0.05, BCE and 1,000 epochs. Final metrics use the model after all updates. Final BCE/accuracy: Sigmoid 0.693194 / 29.30%; ReLU 0.138952 / 100.00%. No claim of generalization is made from training accuracy.

## Requirement comparison and contents

The supplied submission-page checklist and the README submission checklist agree: three completed Python scripts, `plots/activation_comprehensive_analysis.png`, and a short PDF answering the two questions.

Additional requirements found in the folder:

- README Part 1: observe loss decreasing across five iterations.
- README Part 3: calculate CNN dimensions and parameter count before implementing Conv2d, then include terminal evidence in the report.
- Slide 5: add an image of the Part 1 output table to the PDF.
- Slides 7 and 9: include activation analysis and CNN verification evidence.

The report covers all content above. **Evidence limitation:** system safety restrictions blocked screenshot access to Terminal and Codex. `screenshots/part1_run_transcript.png` and `screenshots/part3_run_transcript.png` are explicitly labelled renderings of actual run-log excerpts, not native screenshots. Full original output is included in `logs/main_run.txt`; `logs/part3_before_implementation.txt` records the unimplemented CNN run, and `logs/part3_hand_calculation.txt` records the independent calculation. If the course strictly requires native terminal screenshots, replace the two transcript images with screenshots from your own terminal run and rebuild the PDF.

The ZIP also includes the unchanged runner `main.py`, original instructions `readme.md`, `report.tex`, dependency pins, validation results and experiment logs. Original `_hand` copies, the slide deck, Python caches and LaTeX intermediate files are not included.

## Report source

The two-page PDF is compiled from LaTeX using the included PNG files:

```sh
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The full-size activation PNG is included separately; its report version is scaled to fit. No name or student ID was invented.
