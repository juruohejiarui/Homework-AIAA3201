# Lab09 submission and requirement audit

The submission package is `../lab_week09_submission.zip`. It contains only the
three completed task scripts, the single-page English report and the three JPG
outputs requested by the submission page. The README's requirement to submit
the output images is covered by those same files under `output/`.

## Preserved rules

- The ten original files are preserved in `../lab_week09_hand/`, with byte/hash
  verification against the original ZIP. Windows backslashes were converted to
  directory separators when extracting the package.
- All original comments, six TODO comments, docstrings and code markers remain.
  Only placeholder expressions were replaced in the three task scripts.
- `main.py`, `utils.py`, the README, input images and original `.gitkeep` are
  unchanged. The supplied inference, visualization and dynamic-prompt logic is
  retained. No dummy images were used.
- Actual pretrained Faster R-CNN, FCN and SAM inference was run. The full-size
  JPG files are unmodified outputs of the supplied visualization code.
- The English report is exactly one A4 page, built using XeLaTeX and visually
  checked after rendering. Its source remains in `report.tex`.
- No unknown name or student number was filled in. No Canvas upload was made.

## Environment separation and restoration

An initial installation in the shared `aiaa3201` environment downgraded
`huggingface-hub` from 2.2.0 to 0.36.2 and introduced seven new packages.
Those additions were removed and `huggingface-hub==2.2.0` was restored.
The restored package inventory matches the pre-installation state reconstructed
from the pip transaction. `pip check` reported no broken requirements and the
Project evaluation CLI import/help check succeeded.

The restored environment was cloned with Conda into `aiaa3201-lab`. The complete
Python package inventory was compared before installing transformers and matched
exactly. `transformers==4.57.1` and its required dependencies were then installed
only in the lab environment. The original environment was rechecked after that
installation and remained unchanged. Restoration and inventory records are in
`logs/project_*` and `logs/lab_environment_before_transformers.json`.

The old Hub client cannot parse the official server's download metadata in this
environment. `prepare_sam_cache.py` downloads the official SAM snapshot directly
and verifies file size and official Git/LFS hashes, without changing task code.
The exact revision and file hashes are recorded in `logs/sam_download.json`.
The separate cache is `~/.cache/huggingface-aiaa3201-lab/`; model weights are not
included in the submission archive.

## Reproduce

From the repository root:

```sh
conda activate aiaa3201-lab
python lab_week09/prepare_sam_cache.py
cd lab_week09
HF_HOME="$HOME/.cache/huggingface-aiaa3201-lab" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/aiaa3201-lab09-mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python main.py
```

The cache-preparation step is idempotent and only needs downloading on first use.
`run_checked.py` runs the official runner and additional checks with these settings.
Python/package versions are recorded in `logs/environment.txt` and
`logs/requirements_tested.txt`.

## Requirement and verification checklist

| Source | Requirement | Evidence |
| --- | --- | --- |
| README Task 1; code TODOs | Pretrained Faster R-CNN; highest-score box with dimensions retained | Real cat detection; unsorted-score selection and empty branch checked |
| README Task 2; code TODOs | Pretrained FCN; ImageNet normalization; class-axis argmax | 2D integer mask at original size, five classes; provided preprocessing unchanged |
| README Task 3; code TODOs | Exact SAM model ID, dynamic box prompt, postprocessing, candidate 0 | Original-size nonempty Boolean mask; prompt compared to Task 3 detector result |
| Submission page; README | Run main.py and generate three images | logs/main_stdout.txt and all three JPGs |
| Submission page; README submission instructions | Submit three scripts and output images | Exact final ZIP member list in logs/final_validation.json |
| Submission page | PDF answers ImageNet normalization and SAM prompting questions | Both answers and three actual result images in report.pdf |
| Previous user rules | One-page English PDF, retained TODOs and original backup | Rendered single page; comment equality and original hashes checked |
| Agreed plan | Unpack final ZIP, supply original runner and inputs, rerun | logs/zip_rerun_stdout.txt |

The report does not claim a measured no-normalization ablation. It explains the
distribution mismatch and possible degradation. The third image contains a
wristwatch, cup and other objects. The detector's highest-confidence prediction
is COCO class 85 (clock) on the wristwatch, so SAM segments that region. This
follows the required highest-score pipeline; the target was not manually forced
to the cup. Candidate 0 is selected as the lab specifies, not by comparing SAM
IoU scores. No ground-truth segmentation accuracy is claimed.

The FCN's fixed 21-class label space includes Pascal VOC foreground categories;
the street's road/buildings can therefore be background. Its random palette
changes between runs, so output JPGs are not required to be byte-identical across
reruns. The three archived JPGs match the formal submitted files exactly.

## Final archive

```text
task1_detection.py
task2_segmentation.py
task3_sam.py
report.pdf
output/task1_detection_result.jpg
output/task2_segmentation_result.jpg
output/task3_sam_real_result.jpg
```

For the extracted rerun, the unchanged original `main.py`, `utils.py` and `input/`
were supplied separately from the backup, as agreed for the minimal submission
archive. Extracted output images were removed before rerunning so that stale
images could not hide a failed task. Logs, raw mask arrays, report sources,
validation helpers, inputs, backups and model weights remain outside the ZIP.
