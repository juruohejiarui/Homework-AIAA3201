"""Run and validate the provided lab runner using the isolated lab environment."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time


def main():
    assert Path(sys.prefix).name == 'aiaa3201-lab', 'Use conda environment aiaa3201-lab'
    folder = Path(__file__).resolve().parent
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR='/tmp/aiaa3201-lab09-mpl',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', HF_HUB_DISABLE_XET='1',
               HF_HOME=str(Path.home()/'.cache/huggingface-aiaa3201-lab'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], env=env,
                                 capture_output=True, text=True, check=True)
    (folder/'logs/requirements_tested.txt').write_text(environment.stdout)
    for script, prefix in [('main.py', 'main'), ('verify_extra.py', 'extra')]:
        started = time.time()
        with (folder/f'logs/{prefix}_stdout.txt').open('w') as out, \
                (folder/f'logs/{prefix}_stderr.txt').open('w') as err:
            run = subprocess.run([sys.executable, '-u', script], cwd=folder, env=env,
                                 stdout=out, stderr=err)
        output = (folder/f'logs/{prefix}_stdout.txt').read_text()
        print(output, flush=True)
        assert run.returncode == 0, (folder/f'logs/{prefix}_stderr.txt').read_text()[-4000:]
        assert 'Error occurred' not in output and '[Pipeline] Error' not in output
        for name in ['task1_detection_result.jpg', 'task2_segmentation_result.jpg',
                     'task3_sam_real_result.jpg']:
            path = folder/'output'/name
            assert path.is_file() and path.stat().st_mtime >= started and path.stat().st_size > 0
        if prefix == 'extra':
            assert 'PASS: all real output types' in output and 'empty-detection branch' in output
        print(prefix, 'verified; elapsed seconds:', round(time.time()-started, 2), flush=True)
    from PIL import Image
    expected = {'task1_detection_result.jpg': (2048, 2048),
                'task2_segmentation_result.jpg': (2752, 1536),
                'task3_sam_real_result.jpg': (2400, 1792)}
    for name, size in expected.items():
        with Image.open(folder/'output'/name) as image:
            assert image.size == size and image.format == 'JPEG'
            image.verify()
    print('All real inference and result-file checks passed.', flush=True)


if __name__ == '__main__':
    main()
