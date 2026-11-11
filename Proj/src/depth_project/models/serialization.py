"""Array-tree transport for the public model-specific dictionaries, without pickle."""
import numpy as np
import torch


def pack(value, arrays):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        dtype = "bfloat16" if tensor.dtype == torch.bfloat16 else None
        array = (tensor.view(torch.uint16) if dtype else tensor).numpy()
        index = len(arrays)
        arrays.append(array if array.flags.c_contiguous else np.ascontiguousarray(array))
        return {"__tensor__": index, "dtype": dtype}
    if isinstance(value, np.ndarray):
        index = len(arrays)
        arrays.append(value if value.flags.c_contiguous else np.ascontiguousarray(value))
        return {"__array__": index}
    if isinstance(value, dict):
        return {"__dict__": [[k, pack(v, arrays)] for k,v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"__tuple__" if isinstance(value, tuple) else "__list__": [pack(v, arrays) for v in value]}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported model transport value {type(value)}")


def unpack(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"__dict__"}:
            return {k: unpack(v, arrays) for k,v in value["__dict__"]}
        if set(value) == {"__tuple__"}:
            return tuple(unpack(v, arrays) for v in value["__tuple__"])
        if set(value) == {"__list__"}:
            return [unpack(v, arrays) for v in value["__list__"]]
        if set(value) == {"__tensor__", "dtype"}:
            tensor = torch.from_numpy(arrays[value["__tensor__"]])
            return tensor.view(torch.bfloat16) if value["dtype"] == "bfloat16" else tensor
        if set(value) == {"__array__"}:
            return arrays[value["__array__"]]
        return {k: unpack(v, arrays) for k,v in value.items()}
    if isinstance(value, list):
        return [unpack(v, arrays) for v in value]
    return value
