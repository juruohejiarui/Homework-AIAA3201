"""Extract finished KITTI recovery downloads and check their folder layout."""
import argparse
from collections import Counter
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
from zipfile import BadZipFile, ZipFile

from depth_project.paths import DATASETS, PROJECT
from depth_project.storage.mount import require_dataset_volume


def archive_layout(archive):
    drive = archive.stem
    prefix = (drive[:10], drive)
    directories = {Path('.')}
    files = {}
    with ZipFile(archive) as source:
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or '..' in path.parts or path.parts[:2] != prefix:
                raise ValueError(f'Unexpected ZIP path: {entry.filename}')
            relative = Path(*path.parts[2:])
            if entry.is_dir():
                directories.add(relative)
            else:
                files[relative] = entry.file_size
            directories.update(relative.parents)
    return directories, files


def check_layout(target, directories, files):
    if not target.is_dir():
        return False, '目录不存在'
    actual_directories = {Path('.')}
    actual_files = {}

    def raise_error(error):
        raise error

    try:
        for parent, children, names in os.walk(target, onerror=raise_error):
            relative = Path(parent).relative_to(target)
            actual_directories.update(relative / name for name in children)
            for name in names:
                path = Path(parent) / name
                actual_files[relative / name] = path.stat().st_size
    except OSError as error:
        return False, f'目录访问失败：{error}'
    if actual_directories != directories:
        return False, '子目录集合与 ZIP 不一致'
    if actual_files != files:
        return False, '文件路径或大小与 ZIP 不一致'
    return True, f'{len(files)}个文件，目录与大小一致'


def another_writer(archive):
    commands = subprocess.check_output(['ps', '-eo', 'comm=,args='], text=True)
    for line in commands.splitlines():
        command, _, arguments = line.strip().partition(' ')
        if command in {'wget', 'curl', 'unzip', '7z'} and archive.name in arguments:
            return True
    return False


def process_archive(archive, raw, check_only=False):
    if not archive.is_file():
        print(f'[跳过] {archive.name}：尚未下载', flush=True)
        return 'skipped'
    if archive.with_name(archive.name + '.aria2').exists() or another_writer(archive):
        print(f'[跳过] {archive.name}：仍在下载或被其他解压任务使用', flush=True)
        return 'skipped'
    try:
        directories, files = archive_layout(archive)
    except BadZipFile:
        print(f'[跳过] {archive.name}：ZIP 尚不完整或已损坏', flush=True)
        return 'skipped'

    target = raw / archive.stem[:10] / archive.stem
    correct, description = check_layout(target, directories, files)
    if correct:
        print(f'[通过] {archive.stem}：{description}，无需重复解压', flush=True)
        return 'passed'
    if check_only:
        print(f'[待处理] {archive.stem}：{description}', flush=True)
        return 'pending'

    print(f'[解压] {archive.stem}：{description}', flush=True)
    if target.exists():
        backup = raw.parent / 'recovery_backups' / f'{archive.stem}.{time.time_ns()}'
        backup.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the old tree; never delete it, even if unzip fails.
        target.rename(backup)
        if target.exists() or not backup.is_dir():
            raise RuntimeError(f'旧目录备份状态异常，停止操作：{backup}')
        print(f'[备份] {backup}', flush=True)

    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(['unzip', '-q', '-o', str(archive), '-d', str(raw)], check=True)

    correct, description = check_layout(target, directories, files)
    if not correct:
        raise RuntimeError(f'{target}：{description}')
    print(f'[通过] {archive.stem}：{description}', flush=True)
    return 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='只对照目录和文件大小，不解压')
    args = parser.parse_args()
    root = (DATASETS / 'KITTI').resolve()
    if not args.check_only:
        require_dataset_volume()
    lines = (PROJECT / 'storage/kitti_repair_urls.txt').read_text().splitlines()
    names = [line.strip().rsplit('/', 1)[-1] for line in lines
             if line.strip() and not line.lstrip().startswith('#')]
    results = Counter()
    for name in dict.fromkeys(names):
        archive = root / 'archives/recovery' / name
        try:
            status = process_archive(archive, root / 'raw', args.check_only)
            results[status] += 1
        except (OSError, ValueError, RuntimeError, BadZipFile, subprocess.CalledProcessError) as error:
            results['failed'] += 1
            print(f'[失败] {name}：{error}；原包和备份保留', flush=True)
    print(f'本轮结果：{dict(results)}', flush=True)
    if results['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
