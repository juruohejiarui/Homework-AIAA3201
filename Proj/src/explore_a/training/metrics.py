"""Detached current-batch depth scores using the benchmark mask and clipping limits."""
import torch

from explore_a.data.reading import depth_range


@torch.no_grad()
def add_depth_scores(summary, output, data, rows, config):
    from explore_a.frontend import relative
    if relative(config):
        from explore_a.evaluation.depth import field_scores
        for i, row in enumerate(rows):
            valid = data["gt_valid"][i].bool()
            summary.add(row, field_scores(output["depth"][i], data["gt"][i], valid,
                                          config, row["dataset"]), valid.sum())
        return
    # Reduce on the prediction device; transfer only seven scalars per valid frame.
    for local, row in enumerate(rows):
        valid = data["gt_valid"][local].bool()
        prediction = output["depth"][local].detach().float()
        prediction = prediction.clamp(*depth_range(row["dataset"], config["data"]))[valid]
        target = data["gt"][local].float()[valid]
        if not target.numel():
            summary.add(row, None, 0)
            continue
        error = prediction - target
        ratio = torch.maximum(prediction / target, target / prediction)
        values = torch.stack([
            (error.abs() / target).mean(), (error.square() / target).mean(),
            error.square().mean().sqrt(),
            (prediction.log() - target.log()).square().mean().sqrt(),
            (ratio < 1.25).float().mean(), (ratio < 1.25 ** 2).float().mean(),
            (ratio < 1.25 ** 3).float().mean(),
        ]).cpu().tolist()
        names = ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")
        summary.add(row, dict(zip(names, values)), target.numel())


def finish_depth_scores(summary):
    if any(count["valid_frames"] for count in summary.counts.values()):
        return summary.finish()
    return None
