"""Fixed development frame-pair subsets; retain sequences and original pair semantics."""
from collections import defaultdict
from hashlib import blake2b


def select_dev(rows, fraction=1., seed=2026):
    if not 0 < fraction <= 1:
        raise ValueError("training.dev_sample_fraction must be in (0, 1]")
    if fraction == 1 or not rows:
        return rows
    sources = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sources[row["dataset"]][row["sequence"]].append(row)
    selected = set()
    for source, groups in sorted(sources.items()):
        target = max(1, round(sum(len(group) for group in groups.values()) * fraction))
        # Keep every sequence represented when the source budget permits it.
        minimum = int(target >= len(groups))
        ideal = {name: len(group) * fraction for name, group in groups.items()}
        quotas = {name: max(minimum, int(ideal[name])) for name in groups}
        while sum(quotas.values()) > target:
            name = max((n for n in groups if quotas[n] > minimum),
                       key=lambda n: (quotas[n] - ideal[n], n))
            quotas[name] -= 1
        while sum(quotas.values()) < target:
            name = max((n for n in groups if quotas[n] < len(groups[n])),
                       key=lambda n: (ideal[n] - quotas[n], n))
            quotas[name] += 1
        for name, group in groups.items():
            ranked = sorted(group, key=lambda row: (
                blake2b(f"dev:{seed}:{source}:{row['id']}".encode(), digest_size=8).digest(),
                row["id"]))
            selected.update(row["id"] for row in ranked[:quotas[name]])
    # Keep full original pairs and their order; never turn sparse selections into new pairs.
    return [row for row in rows if row["id"] in selected]
