"""Validate real outputs while running the unchanged lab runner a second time."""
from pathlib import Path
import json
import runpy

import numpy as np
import torch

import task1_detection as detection
import task2_segmentation as segmentation
import task3_sam as sam


def main():
    results = {"detections": []}
    original_detection = detection.run_object_detection
    original_segmentation = segmentation.run_semantic_segmentation
    original_sam = sam.run_sam_zeroshot

    def checked_detection(image):
        boxes, labels, scores = original_detection(image)
        assert boxes.shape == (1, 4) and labels.shape == scores.shape == (1,)
        assert all(isinstance(x, np.ndarray) for x in (boxes, labels, scores))
        assert np.isfinite(boxes).all() and np.isfinite(scores).all()
        assert np.issubdtype(labels.dtype, np.integer) and 0 <= scores[0] <= 1
        x1, y1, x2, y2 = boxes[0]
        height, width = image.shape[:2]
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        results["detections"].append({"image_shape": list(image.shape),
                                      "box": boxes[0].tolist(), "label": int(labels[0]),
                                      "score": float(scores[0])})
        return boxes, labels, scores

    def checked_segmentation(image):
        mask = original_segmentation(image)
        assert isinstance(mask, np.ndarray) and mask.shape == image.shape[:2]
        assert np.issubdtype(mask.dtype, np.integer)
        assert mask.min() >= 0 and mask.max() <= 20
        ids, counts = np.unique(mask, return_counts=True)
        assert len(ids) > 1
        results["segmentation"] = {"shape": list(mask.shape),
                                   "class_pixels": {str(i): int(c) for i, c in zip(ids, counts)}}
        np.save('validation/semantic_mask.npy', mask, allow_pickle=False)
        return mask

    def checked_sam(image, prompt):
        np.testing.assert_array_equal(np.asarray(prompt), results["detections"][-1]["box"])
        mask = original_sam(image, prompt)
        assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_
        assert mask.shape == image.shape[:2] and mask.any() and not mask.all()
        results["sam"] = {"shape": list(mask.shape), "foreground_pixels": int(mask.sum()),
                          "foreground_fraction": float(mask.mean()),
                          "prompt": np.asarray(prompt).tolist(), "candidate_index": 0}
        np.save('validation/sam_mask.npy', mask, allow_pickle=False)
        return mask

    detection.run_object_detection = checked_detection
    segmentation.run_semantic_segmentation = checked_segmentation
    sam.run_sam_zeroshot = checked_sam
    runpy.run_path('main.py', run_name='__main__')
    assert len(results["detections"]) == 2 and 'segmentation' in results and 'sam' in results
    Path('logs/real_output_validation.json').write_text(json.dumps(results, indent=2) + '\n')
    print('PASS: all real output types, dimensions, values and dynamic SAM prompt verified.')

    # An unsorted synthetic output checks selection logic, not neural inference quality.
    constructor = detection.torchvision.models.detection.fasterrcnn_resnet50_fpn
    class StubDetector:
        def eval(self):
            return self

        def __call__(self, images):
            return [{"boxes": torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]),
                     "labels": torch.tensor([17, 47]), "scores": torch.tensor([0.2, 0.9])}]

    try:
        detection.torchvision.models.detection.fasterrcnn_resnet50_fpn = lambda **kw: StubDetector()
        boxes, labels, scores = original_detection(np.zeros((10, 10, 3), dtype=np.uint8))
        np.testing.assert_array_equal(boxes, [[5., 6., 7., 8.]])
        assert labels.tolist() == [47] and np.isclose(scores[0], 0.9)
        class EmptyDetector(StubDetector):
            def __call__(self, images):
                return [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long),
                         "scores": torch.empty(0)}]
        detection.torchvision.models.detection.fasterrcnn_resnet50_fpn = lambda **kw: EmptyDetector()
        assert all(x.size == 0 for x in original_detection(np.zeros((10, 10, 3), dtype=np.uint8)))
    finally:
        detection.torchvision.models.detection.fasterrcnn_resnet50_fpn = constructor
    print('PASS: highest-score selection with unsorted scores and the empty-detection branch.')


if __name__ == '__main__':
    main()
