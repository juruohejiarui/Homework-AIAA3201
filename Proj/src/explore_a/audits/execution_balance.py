"""Separate training and full-inference capacity models; no probability grid search."""
import math

from scipy.optimize import linprog


def solve(local_frozen, cached, remote_frozen, head_train, head_eval, remote_full,
          availability=1., scoring=0.):
    costs = (local_frozen, cached, remote_frozen, head_train, head_eval, remote_full)
    if any(not math.isfinite(x) or x < 0 for x in costs) or not 0 <= availability <= 1:
        raise ValueError("Invalid measured execution costs")

    def optimize(training, remote_fixed=None):
        h = head_train if training else head_eval
        remote = remote_frozen if training else remote_full
        # Variables local/cache/remote fractions and bottleneck seconds per pair.
        host = [local_frozen+h, h, h if training else 0., -1.]
        bounds = [(0,1), (0,availability), (0,1), (scoring,None)]
        if remote_fixed is not None:
            bounds[2] = (remote_fixed,remote_fixed)
        result = linprog([0,0,0,1], A_ub=[host,[0,cached,0,-1],[0,0,remote,-1]],
                         b_ub=[0,0,0], A_eq=[[1,1,1,0]], b_eq=[1], bounds=bounds, method="highs")
        if not result.success:
            raise ValueError(result.message)
        return dict(zip(("local","cache","remote","pair_s"),map(float,result.x)))

    results = {}
    for training in (True,False):
        theory = optimize(training)
        conservative = optimize(training, math.floor((theory["remote"]+1e-10)*10)/10)
        conservative["cache"] = math.floor((conservative["cache"]+1e-10)*100)/100
        conservative["local"] = 1-conservative["cache"]-conservative["remote"]
        local, cache, remote = (conservative[k] for k in ("local", "cache", "remote"))
        h = head_train if training else head_eval
        conservative["pair_s"] = max(local*(local_frozen+h) + cache*h + (remote*h if training else 0.),
                                     cache*cached, remote*(remote_frozen if training else remote_full), scoring)
        results["train" if training else "evaluation"] = {"theoretical": theory, "conservative":conservative}
    return results
