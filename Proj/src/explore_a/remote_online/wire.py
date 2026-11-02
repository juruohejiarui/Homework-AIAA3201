"""Length-delimited JSON and uncompressed arrays; no pickle or temporary files."""
import json
import struct

import numpy as np

VERSION = "frozen-components-da2-v3"
MAX_BYTES = 128 * 1024 * 1024


class IdentityError(ValueError):
    pass


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        part = stream.read(size - len(data))
        if not part:
            raise EOFError("Remote inference stream closed")
        data.extend(part)
    return data


def write_all(stream, data):
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if not written:
            raise OSError("Remote inference stream cannot accept data")
        remaining = remaining[written:]


def send(stream, metadata, arrays=()):
    metadata = dict(metadata)
    metadata["arrays"] = [{"shape": list(a.shape), "dtype": a.dtype.str} for a in arrays]
    header = json.dumps(metadata, allow_nan=False).encode()
    total = sum(a.nbytes for a in arrays)
    if len(header) > 1024 * 1024 or total > MAX_BYTES:
        raise ValueError("Remote message exceeds memory limit")
    write_all(stream, struct.pack("!QQ", len(header), total))
    write_all(stream, header)
    for array in arrays:
        write_all(stream, np.ascontiguousarray(array).tobytes())
    stream.flush()


def receive(stream):
    header_size, size = struct.unpack("!QQ", read_exact(stream, 16))
    if header_size > 1024 * 1024 or size > MAX_BYTES:
        raise ValueError("Invalid remote message size")
    metadata = json.loads(read_exact(stream, header_size))
    payload = read_exact(stream, size)
    arrays, offset = [], 0
    for description in metadata.pop("arrays"):
        shape, dtype = description["shape"], np.dtype(description["dtype"])
        if (dtype not in (np.dtype("uint8"), np.dtype("float32")) or len(shape) > 4
                or any(type(d) is not int or d <= 0 for d in shape)):
            raise ValueError("Invalid remote array layout")
        length = int(np.prod(shape, dtype=object)) * dtype.itemsize
        if length > size - offset:
            raise ValueError("Truncated remote array")
        # Own writable memory before torchvision/torch converts RGB to a tensor.
        arrays.append(np.frombuffer(payload, dtype, count=length // dtype.itemsize,
                                    offset=offset).reshape(shape).copy())
        offset += length
    if offset != size:
        raise ValueError("Unexpected trailing remote payload")
    return metadata, arrays
