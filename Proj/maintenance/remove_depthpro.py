"""Remove retired predictions and sanitize shared receipts; preserve flow/teacher files."""
import argparse
import json
from pathlib import Path
import shutil


def sanitize(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'depthpro' or key.startswith('frames/'):
                continue
            result[key] = sanitize(item)
        if 'sizes' in result and 'bytes' in result:
            result['bytes'] = sum(result['sizes'].values())
        return result
    if isinstance(value, list):
        return [sanitize(item) for item in value
                if not (isinstance(item, str) and item.startswith('frames/'))]
    return value


def clean_cache(root, apply):
    root = Path(root)
    removed = [p for p in (root/'frames', root/'frontends/depthpro') if p.exists()]
    changed = []
    paths = list(root.glob('*.json')) + list((root/'jobs').glob('*/*.json'))
    for path in paths:
        original = json.loads(path.read_text())
        updated = sanitize(original)
        if updated != original:
            changed.append(str(path))
            if apply:
                pending = path.with_suffix('.json.tmp')
                pending.write_text(json.dumps(updated, indent=2)+'\n')
                pending.replace(path)
    for path in (root/'jobs').glob('*/*files.txt'):
        lines = path.read_text().splitlines()
        kept = [line for line in lines if not line.startswith('frames/')]
        if kept != lines:
            changed.append(str(path))
            if apply:
                path.write_text('\n'.join(kept)+'\n')
    # Frozen job configs retain shared-flow weights, but never the retired predictor.
    import yaml
    for path in list(root.glob('*.yaml')) + list((root/'profiles').glob('*/*.yaml')):
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict):
            continue
        value.get('weights', {}).pop('depthpro', None)
        value['frontend'] = {'name':'da2'}
        changed.append(str(path))
        if apply:
            path.write_text(yaml.safe_dump(value, sort_keys=False))
    if apply:
        for path in removed:
            shutil.rmtree(path)
    return {'delete':list(map(str, removed)), 'rewrite_metadata':changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = clean_cache(args.cache_root, False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(plan, indent=2)+'\n')
    if args.apply:
        clean_cache(args.cache_root, True)
    print(json.dumps({'apply':args.apply, 'deleted_paths':plan['delete'],
                      'metadata_files':len(plan['rewrite_metadata'])}))


if __name__ == '__main__':
    main()
