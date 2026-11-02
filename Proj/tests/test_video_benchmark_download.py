"""Small synthetic checks for selection, validation archive roots and download resumption."""
from io import BytesIO
import json
from pathlib import Path
import tarfile

import pytest

from explore_a.download.tum_extract import extract_archive
from explore_a.download import video_benchmarks as cli, video_sources as sources
from explore_a.audits.tum_validation import audit


def make_tar(path, name):
    with tarfile.open(path, 'w:gz') as archive:
        info = tarfile.TarInfo(name)
        info.size = 3
        archive.addfile(info, BytesIO(b'RGB'))


def test_secret_root_normalized_and_conflicts_rejected(tmp_path):
    archive = tmp_path/'a.tgz'
    name = 'rgbd_dataset_freiburg1_xyz_validation'
    make_tar(archive, 'rgbd_dataset_freiburg1_xyz_secret/rgb/1305033337.072235.png')
    entry = {'name': name+'.tgz', 'bytes': archive.stat().st_size}
    root = tmp_path/'sequences'
    root.mkdir()
    extract_archive(archive, root, entry)
    assert (root/name/'rgb/1305033337.072235.png').read_bytes() == b'RGB'
    extract_archive(archive, root, entry)
    with pytest.raises(ValueError, match='receipt conflict'):
        extract_archive(archive, root, {**entry, 'bytes': entry['bytes']+1})


def test_archive_traversal_never_writes(tmp_path):
    archive = tmp_path/'a.tgz'
    name = 'rgbd_dataset_freiburg3_xyz_validation'
    make_tar(archive, name+'/../../outside')
    with pytest.raises(ValueError, match='Unsafe'):
        extract_archive(archive, tmp_path/'sequences',
                        {'name': name+'.tgz', 'bytes': archive.stat().st_size})
    assert not (tmp_path/'outside').exists()


def test_spring_minimal_assets_and_official_digests(monkeypatch):
    names = ['train_cam_data.zip', 'train_frame_left.zip', 'train_disp1_left.zip',
             'train_flow_FW_right.zip']
    files = [{'dataFile': {'filename': name, 'filesize': 123, 'id': i,
                          'checksum': {'type': 'MD5', 'value': 'a'*32}}}
             for i, name in enumerate(names)]
    payload = {'data': {'latestVersion': {'files': files, 'versionNumber': 2,
                                         'versionMinorNumber': 0}}}
    monkeypatch.setattr(sources, 'metadata', lambda url: json.dumps(payload))
    selected, protocol = sources.spring_files()
    assert {e['name'] for e in selected} == set(names[:3])
    assert all(e['checksum']['value'] == 'a'*32 for e in selected)
    assert protocol['publisher_version'] == [2, 0]


def test_scannet_explicit_validation_scenes(tmp_path, monkeypatch):
    path = tmp_path/'scenes.txt'
    path.write_text('scene0001_00\n')
    monkeypatch.setattr(sources, 'metadata', lambda url: 'scene0001_00\nscene0002_00\n')
    monkeypatch.setattr(sources, 'sized', lambda item: {**item, 'bytes': 123})
    rows, _ = sources.scannet_files(path)
    assert len(rows) == 1 and rows[0]['path'].endswith('scene0001_00.sens')
    path.write_text('scene9999_00\n')
    with pytest.raises(ValueError, match='official v2 validation'):
        sources.scannet_files(path)


def fixture_plan(tmp_path):
    entry = {'path': 'archives/x.tgz', 'url': 'https://cvg.cit.tum.de/x.tgz', 'bytes': 3}
    return {'schema': 1, 'dataset': 'tum-validation', 'output': str(tmp_path/'data'),
            'files': [entry], 'total_bytes': 3}


@pytest.mark.parametrize('field,value', [('path', '../escape'), ('url', 'https://evil.test/x')])
def test_plan_rejects_untrusted_paths(tmp_path, field, value):
    plan = fixture_plan(tmp_path)
    plan['files'][0][field] = value
    with pytest.raises(ValueError):
        cli.validate(plan)


def test_download_resume_without_network(tmp_path, monkeypatch):
    plan = fixture_plan(tmp_path)
    root = cli.validate(plan)
    manifest = tmp_path/'plan.json'
    calls = []
    def transfer(command, **kwargs):
        calls.append(command)
        part = root/'archives/x.tgz.part'
        part.write_bytes(b'RGB')
    monkeypatch.setattr(cli.subprocess, 'run', transfer)
    cli.download(manifest, plan, root, 1, False)
    cli.download(manifest, plan, root, 1, False)
    assert len(calls) == 1
    assert (root/'archives/x.tgz').read_bytes() == b'RGB'
    with pytest.raises(ValueError, match='bundles train'):
        cli.download(manifest, {**plan, 'dataset': 'ddad'}, root, 1, False)


def test_tum_frame_overlap_is_not_decided_by_sequence_name(tmp_path):
    ordinary = tmp_path/'ordinary'
    validation = tmp_path/'validation'
    for root, name in [(ordinary, 'rgbd_dataset_freiburg2_desk'),
                       (validation/'sequences', 'rgbd_dataset_freiburg2_xyz_validation')]:
        for kind in ('rgb', 'depth'):
            folder = root/name/kind
            folder.mkdir(parents=True)
            (folder/'1311868164.363181.png').write_bytes(b'fixture')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'samples': [
        {'dataset': 'tum', 'split': 'train', 'previous_rgb': 'old/1311868164.363181.png',
         'current_rgb': 'old/1311868164.363181.png'}]}))
    result = audit(validation, ordinary, manifest)['sequences'][0]
    assert result['manifest_timestamp_matches']['1311868164.363181.png'] == ['train']
    assert result['ordinary_timestamp_matches']['1311868164.363181.png']
