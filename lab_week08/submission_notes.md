# Lab08 requirement audit and submission status

Status: COMPLETE. Code, plots, final single-page report and final ZIP verified.
The user-supplied genuine terminal screenshot is embedded unchanged in the final
report. The final ZIP was extracted and rerun successfully; regenerated PNGs
are byte-identical to the submitted PNGs. No platform upload was performed.

## Preserved conventions

- `../lab_week08_hand/` contains the three original files, checked byte-for-byte
  before editing. No other lab or Project 3 files are changed.
- All original comments (including all eight TODOs), docstrings and code markers
  remain. Only the eight designated implementation blocks changed.
- Official function signatures, test code, plot parameters and random seed remain.
- The implementation uses Python 3.8-compatible syntax. The actual tested
  environment is recorded in `logs/environment.txt`; Python 3.8 itself was not run.
- English, A4, exactly one page; XeLaTeX source is retained as `report.tex`.

## Completion and submission checklist: union of all sources

| Source | Requirement | Evidence / status |
| --- | --- | --- |
| README Parts 1-3; slides 3-8 | RNN, scaled attention, sinusoidal encoding | All six mathematical TODO blocks implemented |
| Slides 6 and 8 | Softmax on key axis; log-space frequencies; add encoding | Implemented without changing interfaces |
| README Part 4; slide 9 | Two imshow heatmaps saved to output | Both PNGs generated and visually reviewed |
| README; submission page; slide 2 | Run student_lab.py; all tests pass | Three check marks, no stderr; logs/official_stdout.txt |
| Slide 10 | Verify tensor shapes | Three independent check groups passed; logs/extra_stdout.txt |
| README; submission page | PDF answers both specified questions | Both answers present in report.pdf |
| Slide 10 | PDF includes both generated heatmaps | Embedded in the one-page final report |
| Slide 10 | 1-2 paragraphs analyzing scaling and PE patterns | Two analytical paragraphs, with equations, in final report |
| Slide 10 | PDF includes terminal screenshot showing passes | Embedded user-provided genuine screenshot |
| Prior user convention | PDF no more than one page | Final exactly one page; rendered and reviewed |
| README; submission page; slide 10 | ZIP with completed script, PDF, output PNGs | ../lab_week08_submission.zip verified |
| Submission page; slide 10 | Submit before deadline; no late upload | User handles platform upload; date not supplied |

The slide's `--run_tests` spelling is illustrative: the supplied script runs
`run_tests()` unconditionally under its main guard. The official command is
`python3 student_lab.py`, with no interface change needed.

## Reproduce the completed run

From the repository root:

```sh
conda activate aiaa3201
cd lab_week08
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/aiaa3201-lab08-mpl python3 student_lab.py
python3 verify_extra.py
```

## Final evidence

- `screenshots/verification.png`: genuine user-supplied terminal screenshot;
  shows all three passes, both saved paths and successful completion.
- `report.pdf`: exactly one A4 page with both analytical answers, both generated
  heatmaps and the screenshot; text and rendered layout reviewed.
- `logs/report_compile.log`: successful XeLaTeX compilation, no overfull boxes.
- `logs/final_zip_rerun.txt`: actual output from rerunning the extracted final ZIP.
- `logs/validation.json`: final checks, archive members and SHA-256 hash.
- `../lab_week08_submission.zip`: exactly the completed script, final PDF and
  the two PNGs under `output/`. All member bytes match the formal files.

The earlier `report_draft.pdf` is retained locally as a historical draft only.

Original backups, slides, draft report, source, supplemental checks, logs, caches
and TeX intermediate files stay outside the submission ZIP. No unknown name or
student number is filled in, and nothing is uploaded automatically.
