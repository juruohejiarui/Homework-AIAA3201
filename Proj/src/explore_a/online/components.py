"""One frozen component contract shared by local execution, NPZ readers and memory RPC."""
import numpy as np
import torch

from explore_a.data.reading import fit_image
from explore_a.online.lazy import LazyFrozen


def validate(operation, values, canvas, image_shape, config, native=False):
    if operation == 'prediction':
        shapes = {'prediction': tuple(image_shape[:2]) if native else (1, *canvas)}
    elif operation == 'flow':
        shapes = {'forward': (2, *canvas), 'backward': (2, *canvas)}
    elif operation == 'teacher':
        shapes = {'visibility': (1, *canvas)}
    else:
        raise ValueError('Unknown frozen operation: '+operation)
    if set(values) != set(shapes):
        raise ValueError('Frozen result components differ')
    for key, shape in shapes.items():
        value = values[key]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            detail = f'{value.shape}/{value.dtype}'
            raise ValueError(f'Invalid frozen result {operation}/{key}: {detail}')
        if key == 'prediction' and (value < 0).any():
            raise ValueError('Invalid negative/nonpositive frozen prediction')
    return values


class LocalExecutor:
    """Called only by the main GPU thread, including while another task waits on DGX."""
    def __init__(self, config, device='cuda'):
        self.config, self.frozen = config, LazyFrozen(config, device)

    @torch.no_grad()
    def execute(self, operation, images, canvas, *, flow=None, native=False):
        if operation == 'prediction':
            prediction = self.frozen.depth(images[0])
            value = prediction if native else fit_image(prediction, canvas)[0][None]
            values = {'prediction':value}
        elif operation == 'flow':
            forward, backward = self.frozen.flow(flow, images[0], images[1])
            values = {'forward': forward, 'backward': backward}
        elif operation == 'teacher':
            values = {'visibility': self.frozen.visibility(images[0], images[1], images[2])}
        else:
            raise ValueError(operation)
        values = {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}
        return validate(operation, values, canvas, images[0].shape, self.config, native)
