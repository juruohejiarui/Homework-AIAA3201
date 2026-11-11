"""Single-frame batching retains values, axes and gradients without a CPU stack copy."""
import numpy as np
import pytest
import torch

from depth_project.models.api import ImageAdapter, ModelInput


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("tensor_output", [False, True])
def test_image_batch_values_and_gradients(count, tensor_output):
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6).requires_grad_()

    class Adapter(ImageAdapter):
        kind = "relative_inverse"

        def predict(self, rgb):
            return value if tensor_output else value.detach().numpy()

    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    request = ModelInput(rgb=rgb if count == 1 else [rgb]*count,
                         frame_ids=tuple(range(count)))
    result = Adapter().forward(request)
    torch.testing.assert_close(result.prediction, value.unsqueeze(0).expand(count, -1, -1),
                               rtol=0, atol=0)
    assert result.frame_ids == request.frame_ids
    assert result.representation == "relative_inverse"
    if count == 1:
        assert result.prediction.data_ptr() == value.data_ptr()
    if tensor_output:
        result.prediction.sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, count))
