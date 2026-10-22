"""Decode a video into RGB frames with explicit source frame indices."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def prepare_video(source, output, fps=10, max_resolution=640, max_frames=300):
    capture = cv2.VideoCapture(str(source))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        capture.release()
        raise OSError(f"Cannot read video: {source}")
    stride = max(1, round(source_fps / fps))
    frames = []
    indices = []
    index = 0
    try:
        while len(frames) < max_frames:
            decoded, image = capture.read()
            if not decoded:
                break
            if index % stride == 0:
                height, width = image.shape[:2]
                scale = min(1, max_resolution / max(height, width))
                size = (max(2, int(width * scale) // 2 * 2),
                        max(2, int(height * scale) // 2 * 2))
                image = cv2.resize(image, size)
                frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                indices.append(index)
            index += 1
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("At least two decoded frames are required")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, frames=np.stack(frames), fps=source_fps / stride,
                        source_indices=np.array(indices), source=str(Path(source).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--max-resolution", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    prepare_video(args.video, args.output, args.fps, args.max_resolution, args.max_frames)
    print(args.output)


if __name__ == "__main__":
    main()
