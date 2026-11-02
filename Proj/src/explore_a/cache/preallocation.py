"""Native Linux disk reservation, without libc's zero-writing allocation emulation."""
import ctypes
import errno
import os
import sys


def _native_allocator():
    if sys.platform != "linux":
        return None
    function = getattr(ctypes.CDLL(None, use_errno=True), "fallocate64", None)
    if function is not None:
        function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong)
        function.restype = ctypes.c_int
    return function


_fallocate = _native_allocator()


def reserve(fd, size):
    """Reserve exactly size bytes before writing; unsupported filesystems simply skip it.

    Unlike posix_fallocate, fallocate does not emulate unsupported reservations by
    writing zeros. Space and I/O failures must propagate, rather than publish a cache
    file whose reservation failed. Allocation does not guarantee a single extent.
    """
    if _fallocate is None or size <= 0:
        return False
    while _fallocate(fd, 0, 0, size) != 0:
        error = ctypes.get_errno()
        if error == errno.EINTR:
            continue
        if error in (errno.ENOSYS, errno.EOPNOTSUPP):
            return False
        raise OSError(error, os.strerror(error))
    return True
