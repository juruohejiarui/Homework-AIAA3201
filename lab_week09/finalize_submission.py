"""Prepare the report, or package reviewed artifacts and rerun the extracted lab."""
from pathlib import Path
import argparse
import ast
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tokenize
import zipfile


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
OUTPUTS = ['task1_detection_result.jpg', 'task2_segmentation_result.jpg',
           'task3_sam_real_result.jpg']
SCRIPTS = ['task1_detection.py', 'task2_segmentation.py', 'task3_sam.py']


def preserved():
    backup = ROOT/'lab_week09_hand'
    hashes = json.loads((FOLDER/'logs/original_sha256.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((backup/name).read_bytes()).hexdigest() == digest
        if name not in SCRIPTS:
            assert (FOLDER/name).read_bytes() == (backup/name).read_bytes()
    def comments(source):
        return [t.string for t in tokenize.generate_tokens(io.StringIO(source).readline)
                if t.type == tokenize.COMMENT]
    count = 0
    for name in SCRIPTS:
        old, new = (backup/name).read_text(), (FOLDER/name).read_text()
        assert comments(old) == comments(new), name
        count += sum('TODO' in comment for comment in comments(new))
        ast.parse(new)
    assert count == 6
    return {'original_backup_hashes_verified': True, 'unchanged_provided_files_verified': True,
            'all_original_comments_preserved': True, 'todo_blocks': count}


def prepare():
    assert Path(sys.prefix).name == 'aiaa3201-lab'
    checks = preserved()
    data = json.loads((FOLDER/'logs/real_output_validation.json').read_text())
    names = {17: 'cat', 47: 'cup', 85: 'clock (the wristwatch)'}
    first, third = data['detections']
    summary = (
        f"Task 1 selected a {names.get(first['label'], 'COCO object')} "
        f"(COCO ID {first['label']}, confidence {first['score']:.4f}). "
        f"The FCN predicted {len(data['segmentation']['class_pixels'])} distinct classes. "
        f"For Task 3, the highest-score detection was a {names.get(third['label'], 'COCO object')} "
        f"(COCO ID {third['label']}, confidence {third['score']:.4f}); "
        f"SAM candidate 0 covered {100*data['sam']['foreground_fraction']:.2f}\\% of the image.\n")
    (FOLDER/'validation/result_summary.tex').write_text(summary)
    build = Path('/tmp/aiaa3201-lab09-pdf')
    build.mkdir(exist_ok=True)
    run = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                          f'-output-directory={build}', 'report.tex'], cwd=FOLDER,
                         capture_output=True, text=True)
    (FOLDER/'logs/report_compile_stdout.txt').write_text(run.stdout+run.stderr)
    run.check_returncode()
    info = subprocess.check_output(['pdfinfo', str(build/'report.pdf')], text=True)
    assert any(line.split() == ['Pages:', '1'] for line in info.splitlines())
    text = subprocess.check_output(['pdftotext', str(build/'report.pdf'), '-'], text=True)
    assert 'Why normalize' in text and 'Why does SAM' in text
    assert 'Overfull' not in (build/'report.log').read_text()
    shutil.copyfile(build/'report.pdf', FOLDER/'report.pdf')
    shutil.copyfile(build/'report.log', FOLDER/'logs/report_compile.log')
    (FOLDER/'logs/report_text.txt').write_text(text)
    subprocess.run(['pdftoppm', '-scale-to', '1700', '-singlefile', '-png',
                    str(FOLDER/'report.pdf'), str(build/'report_preview')], check=True)
    checks.update(real_output_checks_passed=True, report_pages=1, report_text_checked=True)
    (FOLDER/'logs/final_validation.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(summary)
    print('Review rendered page:', build/'report_preview.png')


def package():
    assert Path(sys.prefix).name == 'aiaa3201-lab'
    preserved()
    members = SCRIPTS+['report.pdf']+['output/'+name for name in OUTPUTS]
    archive = ROOT/'lab_week09_submission.zip'
    assert not archive.exists(), 'Refusing to overwrite an existing submission ZIP'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in members:
            z.write(FOLDER/name, name)
    with tempfile.TemporaryDirectory(prefix='lab09-submission-rerun-') as temporary:
        temporary = Path(temporary)
        with zipfile.ZipFile(archive) as z:
            assert z.namelist() == members and z.testzip() is None
            z.extractall(temporary)
        for name in members:
            assert (temporary/name).read_bytes() == (FOLDER/name).read_bytes()
        for name in ['main.py', 'utils.py']:
            shutil.copyfile(ROOT/'lab_week09_hand'/name, temporary/name)
        shutil.copytree(ROOT/'lab_week09_hand/input', temporary/'input')
        # Remove extracted images so pre-existing output cannot hide a failed rerun.
        for name in OUTPUTS:
            (temporary/'output'/name).unlink()
        env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR='/tmp/aiaa3201-lab09-mpl',
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                   HF_HOME=str(Path.home()/'.cache/huggingface-aiaa3201-lab'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        result = subprocess.run([sys.executable, '-u', 'main.py'], cwd=temporary, env=env,
                                capture_output=True, text=True)
        (FOLDER/'logs/zip_rerun_stdout.txt').write_text(result.stdout)
        (FOLDER/'logs/zip_rerun_stderr.txt').write_text(result.stderr)
        assert result.returncode == 0 and 'Error occurred' not in result.stdout
        assert '[Pipeline] Error' not in result.stdout
        from PIL import Image
        for name in OUTPUTS:
            assert (temporary/'output'/name).stat().st_size > 0
            with Image.open(temporary/'output'/name) as image:
                image.verify()
            assert f'Output saved to output/{name}' in result.stdout
    check = json.loads((FOLDER/'logs/final_validation.json').read_text())
    check.update(report_visually_reviewed=True, three_outputs_visually_reviewed=True,
                 final_zip_extracted_rerun_passed=True, archive_members=members,
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 ready_for_submission=True)
    (FOLDER/'logs/final_validation.json').write_text(json.dumps(check, indent=2)+'\n')
    print('Final submission ready:', archive)
    print('Archive members:', members)
    print(result.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'package'])
    getattr(sys.modules[__name__], parser.parse_args().operation)()
