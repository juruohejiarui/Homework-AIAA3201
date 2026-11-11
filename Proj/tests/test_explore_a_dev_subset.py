"""Development reduction preserves source budgets, sequence coverage and fixed pairs."""
import random

import pytest

from explore_a.training.dev_subset import select_dev


def test_exact_source_budgets_coverage_and_determinism():
    rows = [{"id": f"{source}:{sequence}:{i}", "dataset": source, "sequence": str(sequence),
             "previous_id": i-1, "current_id": i, "split": "dev"}
            for source in ("kitti", "tum", "nyuv2")
            for sequence, length in enumerate((3, 57, 140)) for i in range(length)]
    state = random.getstate()
    selected = select_dev(rows, .1)
    assert len(selected) == 60
    for source, expected in (("kitti", 22), ("tum", 15), ("nyuv2", 23)):
        subset = [r for r in selected if r["dataset"] == source]
        assert len(subset) == expected
        assert {r["sequence"] for r in subset} == {"0", "1", "2"}
    assert all({k: v for k, v in r.items() if k != "evaluation_clip"} in rows for r in selected)
    assert selected == select_dev(rows, .1)
    assert select_dev(rows[::-1], .1) == selected
    assert state == random.getstate()
    assert select_dev(rows, 1) is rows


def test_small_budget_and_invalid_fraction():
    rows = [{"id": str(i), "dataset": "tum", "sequence": str(i)} for i in range(6)]
    assert len(select_dev(rows, .1)) == 1
    with pytest.raises(ValueError):
        select_dev(rows, 0)
