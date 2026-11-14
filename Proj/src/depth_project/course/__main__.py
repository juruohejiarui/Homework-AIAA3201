"""Run, independently verify, and package complete course Parts 1 and 2."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from depth_project.paths import OUTPUTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'verify', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUTS/'course_part1_part2')
    parser.add_argument('--stage', choices=['prepare', 'stereo', 'nyu', 'da2', 'vda',
                                          'eff-unidepthv2', 'eff-da2', 'eff-vda'])
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == 'run' and not args.stage:
        root.mkdir(parents=True, exist_ok=True)
        for stage in ['prepare', 'stereo', 'nyu', 'da2', 'vda',
                      'eff-unidepthv2', 'eff-da2', 'eff-vda']:
            subprocess.run([sys.executable, '-m', 'depth_project.course', 'run',
                            '--output', str(root), '--stage', stage], check=True)
        from depth_project.course.delivery import report, verify
        report(root)
        verify(root)
        return
    import cv2
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.command == 'verify':
        from depth_project.course.delivery import verify
        verify(root)
    elif args.command == 'report':
        from depth_project.course.delivery import report, verify
        report(root)
        verify(root)
    else:
        from depth_project.course import experiments
        if args.stage in ['prepare', 'stereo']:
            getattr(experiments, args.stage)(root)
        else:
            from depth_project.inference.engine import gpu_session
            with gpu_session():
                if args.stage == 'nyu':
                    experiments.nyu(root)
                elif args.stage in ['da2', 'vda']:
                    experiments.video(root, args.stage)
                else:
                    from depth_project.course.efficiency import measure
                    measure(root, args.stage[4:])


if __name__ == '__main__':
    main()
