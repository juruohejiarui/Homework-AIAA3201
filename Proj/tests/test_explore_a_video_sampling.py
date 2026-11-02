"""Source ratios, conditional sqrt video weights and deterministic original-pair sampling."""
from collections import Counter
import random

import pytest

from explore_a.config import load_config
from explore_a.training.sampling import StepBatches


def rows_for_source(source, lengths):
    return [{"dataset": source, "sequence": f"video-{video}", "id": f"{source}:{video}:{pair}",
             "camera": f"camera-{pair % 2}"}
            for video, length in enumerate(lengths) for pair in range(length)]


def test_sqrt_conditional_video_probability_and_pair_uniformity():
    rows = rows_for_source("nyuv2", [4, 16, 36])
    batches = StepBatches(rows, 1, 1, 20000, 0, 2026, sampling_mode="video_sqrt")
    counts, pairs = Counter(), Counter()
    for batch in batches:
        counts[rows[batch[0]]["sequence"]] += 1
        pairs[batch[0]] += 1
    for video, expected in enumerate((1 / 6, 2 / 6, 3 / 6)):
        assert abs(counts[f"video-{video}"] / 20000 - expected) < .015
    # Each stored pair remains equally likely conditional on its video.
    first_video = [pairs[index] for index in range(4)]
    assert max(first_video) - min(first_video) < 150


def test_eight_distinct_videos_same_source_resume_microbatch_and_global_rng():
    rows = sum((rows_for_source(source, range(2, 18))
                for source in ("kitti", "nyuv2", "tum")), [])
    weights = {"kitti": 3, "nyuv2": 3, "tum": 2}
    state = random.getstate()
    full = list(StepBatches(rows, 8, 8, 10000, 0, 2026, weights, "video_sqrt"))
    assert random.getstate() == state
    split = list(StepBatches(rows, 8, 2, 20, 0, 2026, weights, "video_sqrt"))
    assert [sum(split[i:i + 4], []) for i in range(0, len(split), 4)] == full[:20]
    resumed = list(StepBatches(rows, 8, 8, 10000, 9990, 2026, weights, "video_sqrt"))
    assert resumed == full[9990:]
    sources = Counter()
    for batch in full:
        assert len({rows[i]["dataset"] for i in batch}) == 1
        assert len({rows[i]["sequence"] for i in batch}) == 8
        sources[rows[batch[0]]["dataset"]] += 1
    for source, weight in weights.items():
        assert abs(sources[source] / 10000 - weight / 8) < .02


def test_short_source_recycles_videos_only_after_exhaustion_and_old_fix_is_unchanged():
    rows = rows_for_source("tum", [2, 40, 4])
    batch = next(iter(StepBatches(rows, 8, 8, 1, 0, 2026, sampling_mode="video_sqrt")))
    assert len({rows[i]["sequence"] for i in batch[:3]}) == 3
    assert len({rows[i]["sequence"] for i in batch[3:6]}) == 3
    rng = random.Random(2026)
    rng.choice(["tum"])
    expected = rng.choices(list(range(len(rows))), k=8)
    assert next(iter(StepBatches(rows, 8, 8, 1, 0, 2026))) == expected


@pytest.mark.parametrize("name", ["mixed_h5_balanced", "mixed_h5_final"])
def test_current_configs_use_new_distribution(name):
    from pathlib import Path
    path = Path(__file__).parents[1] / f"configs/explore_a/{name}.yaml"
    config = load_config(path)
    assert config["training"]["sampling_mode"] == "video_sqrt"
    assert config["training"]["source_weights"] == {"kitti": 3, "nyuv2": 3, "tum": 2}
