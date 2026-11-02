"""Synthetic NPZ allocation ordering, legacy bytes, unsupported fallback and failed commits."""
import ctypes
import errno
import os
import zipfile

import numpy as np
import pytest

from explore_a.cache import preallocation, storage


def test_native_reservation_allocates_blocks_before_data(tmp_path):
    path = tmp_path / "reservation.tmp"
    size = 1024 * 1024
    with path.open("wb") as output:
        if not preallocation.reserve(output.fileno(), size):
            pytest.skip("Test filesystem has no native reservation support")
        stat = os.fstat(output.fileno())
        assert output.tell() == 0
        assert stat.st_size == size
        assert stat.st_blocks * 512 >= size
    path.unlink()


def test_npz_reserves_exact_final_size_and_keeps_legacy_bytes(tmp_path, monkeypatch):
    path, legacy = tmp_path / "new.npz", tmp_path / "legacy.npz"
    values = {"forward": np.arange(64, dtype=np.float32).reshape(2, 4, 8)[:, :, ::2],
              "backward": np.full((2, 4, 4), -.25, dtype=np.float32)}
    # Generate a reference with the previous streaming codec, including ZIP64 headers.
    with zipfile.ZipFile(legacy, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key, value in values.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, value, allow_pickle=False)
    sizes = []
    native = storage.reserve

    def reserve(fd, size):
        assert not path.exists()
        assert os.fstat(fd).st_size == 0
        sizes.append(size)
        return native(fd, size)

    monkeypatch.setattr(storage, "reserve", reserve)
    storage.write_npz(path, **values)
    assert sizes == [path.stat().st_size]
    assert path.read_bytes() == legacy.read_bytes()
    assert not path.with_suffix(".tmp").exists()
    for key, value in values.items():
        np.testing.assert_array_equal(storage.read_npz(path)[key], value)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None


@pytest.mark.parametrize("error", [errno.EOPNOTSUPP, errno.ENOSYS, None])
def test_unsupported_native_allocation_keeps_writing_npz(tmp_path, monkeypatch, error):
    def unavailable(*args):
        ctypes.set_errno(error)
        return -1

    monkeypatch.setattr(preallocation, "_fallocate", unavailable if error else None)
    path = tmp_path / "fallback.npz"
    value = np.ones((1, 4, 8), dtype=np.float32)
    storage.write_npz(path, depth=value)
    np.testing.assert_array_equal(storage.read_npz(path)["depth"], value)
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("error", [errno.ENOSPC, errno.EIO])
def test_reservation_failure_preserves_existing_cache_and_removes_temp(
        tmp_path, monkeypatch, error):
    path = tmp_path / "existing.npz"
    original = np.ones((1, 4, 8), dtype=np.float32)
    storage.write_npz(path, depth=original)
    before = path.read_bytes()

    def failed(*args):
        ctypes.set_errno(error)
        return -1

    monkeypatch.setattr(preallocation, "_fallocate", failed)
    with pytest.raises(OSError) as caught:
        storage.write_npz(path, depth=original * 2)
    assert caught.value.errno == error
    assert path.read_bytes() == before
    assert not path.with_suffix(".tmp").exists()


def test_interrupted_reservation_retries(tmp_path, monkeypatch):
    calls = []

    def interrupted_then_supported(fd, mode, offset, size):
        calls.append((mode, offset, size))
        if len(calls) == 1:
            ctypes.set_errno(errno.EINTR)
            return -1
        return 0

    monkeypatch.setattr(preallocation, "_fallocate", interrupted_then_supported)
    with (tmp_path / "retry.tmp").open("wb") as output:
        assert preallocation.reserve(output.fileno(), 4096)
    assert calls == [(0, 0, 4096), (0, 0, 4096)]
