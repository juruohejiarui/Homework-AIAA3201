"""GT-free normalization and positive affine alignment of two inverse-depth predictions."""
import torch


@torch.no_grad()
def statistics(value, mask):
    medians, scales = [], []
    for image, valid in zip(value.float(), mask.bool()):
        selected = image[valid & torch.isfinite(image)]
        if selected.numel():
            quantiles = torch.quantile(selected, selected.new_tensor([.1, .5, .9]))
            medians.append(quantiles[1])
            scales.append((quantiles[2] - quantiles[0]).clamp_min(1e-6))
        else:
            medians.append(image.new_zeros(()))
            scales.append(image.new_ones(()))
    return (torch.stack(medians)[:, None, None, None],
            torch.stack(scales)[:, None, None, None])


@torch.no_grad()
def affine(source, target, mask, minimum=64, robust=True):
    """Three Huber IRLS iterations; no GT or camera-motion assumptions enter this fit."""
    aa, bb, accepted = [], [], []
    for source_i, target_i, valid in zip(source.float(), target.float(), mask.bool()):
        valid = valid & torch.isfinite(source_i) & torch.isfinite(target_i)
        x, y = source_i[valid], target_i[valid]
        a, b, good = source_i.new_ones(()), source_i.new_zeros(()), False
        if x.numel() >= minimum:
            weights = torch.ones_like(x)
            for _ in range(3 if robust else 1):
                total = weights.sum().clamp_min(1e-6)
                mx, my = (weights*x).sum()/total, (weights*y).sum()/total
                variance = (weights*(x-mx).square()).sum()/total
                if variance <= 1e-12 * (x.square().mean() + 1):
                    good = False
                    break
                a = (weights*(x-mx)*(y-my)).sum()/total/variance
                b = my-a*mx
                good = bool(torch.isfinite(a) & torch.isfinite(b) & (a > 0))
                if not good:
                    break
                residual = a*x+b-y
                mad = (residual-residual.median()).abs().median().clamp_min(1e-6)
                weights = (1.345*1.4826*mad/residual.abs().clamp_min(1e-6)).clamp_max(1)
        aa.append(a if good else source_i.new_ones(()))
        bb.append(b if good else source_i.new_zeros(()))
        accepted.append(good)
    shape = (-1, 1, 1, 1)
    return (torch.stack(aa).reshape(shape), torch.stack(bb).reshape(shape),
            torch.tensor(accepted, device=source.device).reshape(shape))
