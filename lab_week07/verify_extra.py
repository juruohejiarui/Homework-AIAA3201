"""Additional checks beyond the unchanged official verification script."""
import numpy as np
import torch
from student_lab import ActivationFunctions, Optimizers, Dropout, BatchNorm, setup_transfer_learning


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(3, 4))
    dout = rng.normal(size=x.shape)
    h = 1e-6
    numerical = (ActivationFunctions.sigmoid(x + h) - ActivationFunctions.sigmoid(x - h)) / (2 * h)
    np.testing.assert_allclose(ActivationFunctions.sigmoid_backward(dout, x), dout * numerical, atol=1e-9)
    print('PASS: Sigmoid non-unit upstream gradient matches finite differences.')

    w = np.array([1.2, -0.7, 0.5], dtype=np.float64)
    reference = torch.tensor(w.copy(), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([reference], lr=0.03, betas=(0.8, 0.95), eps=1e-7)
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, 6):
        gradient = rng.normal(size=w.shape)
        reference.grad = torch.tensor(gradient, dtype=torch.float64)
        opt.step()
        w, m, v = Optimizers.adam(w, gradient, m, v, t, lr=0.03, beta1=0.8, beta2=0.95)
        np.testing.assert_allclose(w, reference.detach().numpy(), atol=1e-12)
        np.testing.assert_allclose(m, opt.state[reference]['exp_avg'].numpy(), atol=1e-12)
        np.testing.assert_allclose(v, opt.state[reference]['exp_avg_sq'].numpy(), atol=1e-12)
    print('PASS: Five Adam steps and uncorrected states match torch.optim.Adam.')

    np.random.seed(2026)
    x = rng.normal(size=(500, 400))
    for p in (0.2, 0.7, 1.0):
        out, mask = Dropout.dropout_forward(x, p=p)
        assert out.shape == x.shape and np.all((mask == 0) | (mask == 1 / p))
        np.testing.assert_array_equal(out, x * mask)
        assert abs(np.mean(mask > 0) - p) < 0.01
        assert abs(mask.mean() - 1) < 0.02
        if p == 1:
            np.testing.assert_array_equal(out, x)
        identity, test_mask = Dropout.dropout_forward(x, p=p, mode='test')
        assert identity is x and test_mask is None
    print('PASS: Dropout p=0.2, 0.7, 1.0; scaling, retention, and test identity.')

    x = rng.normal(size=(7, 4))
    x[:, 3] = 3  # Constant feature checks epsilon protection.
    gamma, beta = rng.normal(size=(2, 4))
    rm, rv = rng.normal(size=4), rng.uniform(0.1, 2, size=4)
    old_rm, old_rv = rm.copy(), rv.copy()
    out, next_rm, next_rv = BatchNorm.batchnorm_forward(x, gamma, beta, rm, rv, momentum=0.6)
    expected = gamma * (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5) + beta
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(next_rm, 0.6 * old_rm + 0.4 * x.mean(0))
    np.testing.assert_allclose(next_rv, 0.6 * old_rv + 0.4 * x.var(0))
    saved_rm, saved_rv = next_rm.copy(), next_rv.copy()
    test_x = x[:1]
    out, rm_test, rv_test = BatchNorm.batchnorm_forward(test_x, gamma, beta, next_rm, next_rv, mode='test')
    np.testing.assert_allclose(out, gamma * (test_x - saved_rm) / np.sqrt(saved_rv + 1e-5) + beta)
    for actual, saved in ((next_rm, saved_rm), (next_rv, saved_rv), (rm_test, saved_rm), (rv_test, saved_rv)):
        np.testing.assert_array_equal(actual, saved)
    print('PASS: BatchNorm affine transform, momentum, constant feature, and fixed test statistics.')

    for classes in (3, 7):
        model = setup_transfer_learning(classes)
        assert all(p.requires_grad == name.startswith('fc.') for name, p in model.named_parameters())
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 64, 64))
        assert out.shape == (2, classes) and torch.isfinite(out).all()
    print('PASS: ResNet18 output shapes and frozen backbone for 3 and 7 classes.')
    print('All 5 additional check groups passed.')


if __name__ == '__main__':
    main()
