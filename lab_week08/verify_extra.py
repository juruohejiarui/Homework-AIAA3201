"""Independent numerical checks; the official tests remain unchanged."""
import math
import torch
from student_lab import VanillaRNNCell, scaled_dot_product_attention, PositionalEncoding


def main():
    torch.manual_seed(2026)
    cell = VanillaRNNCell(3, 5).double()
    x, h = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 5, dtype=torch.float64)
    expected = torch.stack([
        torch.stack([torch.tanh(sum(cell.W_xh.weight[j, i] * row[i] for i in range(3))
                               + sum(cell.W_hh.weight[j, i] * prev[i] for i in range(5)))
                     for j in range(5)]) for row, prev in zip(x, h)
    ])
    actual = cell(x, h)
    assert actual.shape == (4, 5)
    torch.testing.assert_close(actual, expected)
    print('PASS: RNN random rectangular weights match scalar reference.')

    for batch, nq, nk, dk, dv in [(2, 3, 5, 4, 6), (1, 1, 1, 3, 2), (3, 4, 2, 7, 1)]:
        q = torch.randn(batch, nq, dk, dtype=torch.float64)
        k = torch.randn(batch, nk, dk, dtype=torch.float64)
        v = torch.randn(batch, nk, dv, dtype=torch.float64)
        expected_weights = torch.empty(batch, nq, nk, dtype=torch.float64)
        expected_out = torch.empty(batch, nq, dv, dtype=torch.float64)
        for b in range(batch):
            for i in range(nq):
                scores = torch.tensor([sum(float(q[b, i, t] * k[b, j, t]) for t in range(dk))
                                       / math.sqrt(dk) for j in range(nk)], dtype=torch.float64)
                probabilities = torch.exp(scores - scores.max())
                probabilities /= probabilities.sum()
                expected_weights[b, i] = probabilities
                for t in range(dv):
                    expected_out[b, i, t] = sum(probabilities[j] * v[b, j, t] for j in range(nk))
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape == (batch, nq, dv) and weights.shape == (batch, nq, nk)
        assert torch.all(weights >= 0) and torch.isfinite(out).all()
        torch.testing.assert_close(weights.sum(-1), torch.ones(batch, nq, dtype=torch.float64))
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(out, expected_out)
    print('PASS: Attention batch/unequal lengths/value widths, probabilities and scalar reference.')

    for width in (1, 3, 4, 7, 128):
        module = PositionalEncoding(width, max_len=100)
        reference = torch.tensor([[math.sin(pos / 10000 ** (2 * (dim // 2) / width))
                                   if dim % 2 == 0 else
                                   math.cos(pos / 10000 ** (2 * (dim // 2) / width))
                                   for dim in range(width)] for pos in range(100)])
        torch.testing.assert_close(module.pe[0], reference, atol=1e-5, rtol=1e-5)
        inputs = torch.randn(3, 17, width)
        saved = inputs.clone()
        torch.testing.assert_close(module(inputs), inputs + reference[:17].unsqueeze(0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(inputs, saved)
        assert 'pe' in dict(module.named_buffers()) and not list(module.parameters())
        assert 'pe' in module.state_dict()
    print('PASS: Positional encoding nonzero positions, odd/even widths, broadcast, addition and buffer.')
    print('All 3 additional check groups passed.')


if __name__ == '__main__':
    main()
