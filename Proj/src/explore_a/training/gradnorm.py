"""Read-only gradient measurements, not a GradNorm training optimizer."""
from itertools import combinations
import math

import torch

from explore_a.training.losses import fusion_loss
from explore_a.training.prediction import predict


def cosine(left, right):
    denominator = left.norm() * right.norm()
    return float(left.dot(right) / denominator) if denominator > 1e-12 else None


def candidate_weights(losses, initial, norms, alpha=1.5):
    """Relative target-norm candidate; not weights learned by actual GradNorm updates."""
    if any(norms[key] <= 1e-12 or initial[key] <= 1e-12 for key in losses):
        return None
    ratios = {key: losses[key] / initial[key] for key in losses}
    mean = sum(ratios.values()) / len(ratios)
    if mean <= 1e-12:
        return None
    raw = {key: (value / mean) ** alpha / norms[key] for key, value in ratios.items()}
    total = sum(raw.values())
    return {key: len(raw) * value / total for key, value in raw.items()}


def measure(model, batches, config, initial=None):
    blocks = {"encoder": list(model.encoder[-1].parameters()),
              "gate": list(model.gate_head.parameters())}
    parameters = [p for values in blocks.values() for p in values]
    vectors = {name: {} for name in ("depth", "total")}
    losses, terms_by_source = {}, {}
    for source, data in batches.items():
        output, _ = predict(model, data, config, "rule")
        terms = fusion_loss(output, data, config["training"])
        losses[source] = {"depth": float(terms.depth.detach()),
                          "total": float(terms.total.detach())}
        terms_by_source[source] = terms.scalars()
        for name in vectors:
            gradients = torch.autograd.grad(getattr(terms, name), parameters, retain_graph=True,
                                            allow_unused=True)
            split, offset = {}, 0
            for block, values in blocks.items():
                selected = gradients[offset:offset + len(values)]
                vector = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                    for p, g in zip(values, selected)]).detach().float().cpu()
                if not torch.isfinite(vector).all():
                    raise ValueError("Nonfinite diagnostic gradient")
                split[block] = vector
                offset += len(values)
            vectors[name][source] = split
    result = {"losses": losses, "loss_terms": terms_by_source, "objectives": {}}
    for objective, by_source in vectors.items():
        blocks_report = {}
        for block in blocks:
            norms = {source: float(values[block].norm()) for source, values in by_source.items()}
            current = {source: values[objective] for source, values in losses.items()}
            baseline = (current if initial is None else
                        {source: values[objective] for source, values in initial.items()})
            blocks_report[block] = {
                "norms": norms,
                "cosines": {f"{a}:{b}": cosine(by_source[a][block], by_source[b][block])
                            for a, b in combinations(by_source, 2)},
                "candidate_weights": candidate_weights(current, baseline, norms),
                "relative_losses": {s: current[s] / max(baseline[s], 1e-12) for s in current}}
        result["objectives"][objective] = blocks_report
    if not all(math.isfinite(v) for values in losses.values() for v in values.values()):
        raise ValueError("Nonfinite diagnostic loss")
    return result
