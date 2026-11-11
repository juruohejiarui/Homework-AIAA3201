"""CPU-only conditional oracle scoring; GT targets never enter the frozen inference pipeline."""
import numpy as np

from depth_project.evaluation.metrics import depth_metrics
from explore_a.evaluation.alignment import depth, fit, moments, named, ordinal

BRANCHES = ("raw", "half_unaligned", "half_aligned", "oracle_unaligned", "oracle_aligned",
            "oracle_additive10", "oracle_signed_m1_p2", "gt_alignment_diagnostic")


def mapping(current, gt, valid):
    if not valid.any():
        return 1., 0.
    a, b = fit(moments(current[valid], 1/gt[valid].astype(np.float64)))
    if a <= 0:
        a = float(np.median(1/gt[valid])/max(float(np.median(current[valid])), 1e-6))
        b = 0.
    return a, b


def fields(packed, gauge, fraction=.1):
    current, history, raw_history, gt, valid, common, scale = packed
    valid, common = valid.astype(bool), common.astype(bool) & valid.astype(bool)
    inverse = np.where(valid, 1/np.maximum(gt, 1e-6), 0)
    a, b = gauge
    truth = (inverse-b)/max(a, 1e-12)

    def project(lower, upper):
        result = np.maximum(np.maximum(lower, 0), np.minimum(truth, np.maximum(upper, 0)))
        return np.where(common, result, current)

    lo, hi = np.minimum(current, history), np.maximum(current, history)
    rlo, rhi = np.minimum(current, raw_history), np.maximum(current, raw_history)
    first, last = 2*current-history, 2*history-current
    use = common & valid
    ag, bg = fit(moments(raw_history[use], truth[use]))
    good = use.sum() >= 2 and ag > 0
    gh = np.maximum(ag*raw_history+bg, 0) if good else current
    values = dict(zip(BRANCHES, (
        current, np.where(common, .5*(current+raw_history), current),
        np.where(common, .5*(current+history), current), project(rlo, rhi), project(lo, hi),
        project(lo-fraction*scale, hi+fraction*scale),
        project(np.minimum(first, last), np.maximum(first, last)),
        project(np.minimum(current, gh), np.maximum(current, gh)))))
    return values, truth


def fixed_metrics(prediction, gt, mask, gauge, limits, normalizer=None):
    if not mask.any():
        return {}
    inverse = 1/gt[mask].astype(np.float64)
    scale = float(np.median(inverse)) if normalizer is None else normalizer
    error = np.abs((gauge[0]*prediction[mask]+gauge[1]-inverse) / max(scale, 1e-6))
    smooth = np.where(error < .1, .5*error**2/.1, error-.05)
    values = depth_metrics(depth(prediction[mask], gauge, limits), gt[mask])
    return {"inverse_loss": float(smooth.mean()), **named(values, "fixed")}


def score_clip(rows, paths, limits, fraction=.1):
    """Two cheap CPU passes over staging files; frozen models run only during staging."""
    stats = np.zeros(5, np.float64)
    for path in paths:
        p = np.load(path, allow_pickle=False)
        valid = p[4].astype(bool)
        stats += moments(p[0][valid], 1/p[3][valid].astype(np.float64))
    video_gauge = fit(stats)
    fallback = video_gauge[0] <= 0
    if fallback:
        video_gauge = (float(stats[2]/max(stats[1], 1e-6)), 0.) if stats[0] else (1., 0.)
    predicted_stats = {(scope, name): np.zeros(5, np.float64)
                       for scope in ("img", "video") for name in BRANCHES}
    records = []
    for row, path in zip(rows, paths):
        packed = np.load(path, allow_pickle=False)
        current, history, raw_history, gt, valid, common, scale = packed
        valid, common = valid.astype(bool), common.astype(bool) & valid.astype(bool)
        gauges = {"img": mapping(current, gt, valid), "video": video_gauge}
        record = {"id": row["id"], "clip": row.get("evaluation_clip"),
                  "primary": {}, "primary_video": {}, "secondary": {}, "secondary_video": {},
                  "common_pixels": int(common.sum()), "valid_pixels": int(valid.sum())}
        for scope, gauge in gauges.items():
            predictions, truth = fields(packed, gauge, fraction)
            category = "primary" if scope == "img" else "primary_video"
            secondary = "secondary" if scope == "img" else "secondary_video"
            for name, prediction in predictions.items():
                scale_gt = float(np.median(1/gt[valid])) if valid.any() else 1.
                measured = fixed_metrics(prediction, gt, common, gauge, limits, scale_gt)
                measured.update({k+"_full": v for k, v in
                                 fixed_metrics(prediction, gt, valid, gauge, limits).items()})
                record[category][name] = measured or None
                if valid.any():
                    ps = moments(prediction[valid], 1/gt[valid].astype(np.float64))
                    predicted_stats[scope, name] += ps
                    image = depth_metrics(depth(prediction[valid], fit(ps), limits), gt[valid])
                    record[secondary][name] = {
                        **named(image, "img"), **ordinal(prediction[valid], gt[valid], row["id"])}
                else:
                    record[secondary][name] = None
            if common.any():
                losses = {k: v["inverse_loss"] for k, v in record[category].items()}
                if (losses["oracle_aligned"] > losses["raw"]+1e-6 or
                        losses["oracle_additive10"] > losses["oracle_aligned"]+1e-6):
                    raise AssertionError("Conditional oracle nesting violated")
            if scope == "img":
                denom = max(int(common.sum()), 1)
                lo, hi = np.minimum(current, history), np.maximum(current, history)
                projected = predictions["oracle_additive10"]
                touches = ((projected <= np.maximum(lo-fraction*scale, 0)+1e-6) |
                           (projected >= hi+fraction*scale-1e-6))
                record["diagnostics"] = {
                    "rule_coverage": float(common.sum()/max(int(valid.sum()), 1)),
                    "both_high": float(
                        (common & (current > truth) & (history > truth)).sum()/denom),
                    "both_low": float((common & (current < truth) & (history < truth)).sum()/denom),
                    "inside_interval": float((common & (truth >= lo) & (truth <= hi)).sum()/denom),
                    "correction_boundary": float((common & touches).sum()/denom)}
        records.append(record)
    for row, path, record in zip(rows, paths, records):
        packed = np.load(path, allow_pickle=False)
        gt, valid = packed[3], packed[4].astype(bool)
        if not valid.any():
            continue
        gauges = {"img": mapping(packed[0], gt, valid), "video": video_gauge}
        for scope, gauge in gauges.items():
            secondary = "secondary" if scope == "img" else "secondary_video"
            predictions, _ = fields(packed, gauge, fraction)
            for name, prediction in predictions.items():
                values = depth_metrics(depth(prediction[valid], fit(predicted_stats[scope, name]),
                                             limits), gt[valid])
                record[secondary][name].update(named(values, "video"))
    return records, {"scale": video_gauge[0], "shift": video_gauge[1],
                     "positive_ratio_fallback": fallback}
