"""Thin two-endpoint cache commands; workers never run unless the user invokes worker."""
import argparse
import json
from pathlib import Path

from depth_project.paths import PROJECT
from explore_a.cache import remote
from explore_a.cache.settings import DEFAULT_CONFIG, LOCAL_ROOT, RECIPE, REMOTE_ROOT, root_path
from explore_a.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sync-project", "migrate", "plan", "policy", "pipeline",
                                          "worker", "status", "retry", "stop", "remote", "native"))
    parser.add_argument("operation", nargs="?")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--recipe", default=RECIPE)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--from-recipe", default="kitti_tum-fp32-flash-v1")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--host", choices=("local", "dgx", "all"))
    parser.add_argument("--workers", default="dgx", help="local, dgx, or local,dgx")
    parser.add_argument("--scope", choices=("manifest", "benchmark", "nyu"), default="manifest")
    parser.add_argument("--split", choices=("all", "train", "dev", "test"), default="all")
    parser.add_argument("--setting", choices=("short", "long"), default="long")
    parser.add_argument("--datasets", nargs="+", default=("kitti", "sintel", "bonn"))
    parser.add_argument("--benchmark-root", type=Path,
                        default=Path.home()/"Documents/ResourcesLow/Datasets")
    parser.add_argument("--nyu-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.action == "worker" and args.host == "all":
        parser.error("A worker runs on one host: choose local or dgx")
    remote_worker = args.action == "worker" and args.host != "local"
    base = args.base or (REMOTE_ROOT if remote_worker or args.action == "remote" else LOCAL_ROOT)
    root = root_path(base, args.recipe)
    if args.action == "native":
        from explore_a.cache.native import produce
        config = load_config(args.config)
        if args.base:
            config.setdefault("cache", {})["root"] = str(args.base)
        if args.scope == "nyu" and args.nyu_root is None:
            parser.error("--scope nyu requires --nyu-root")
        print(json.dumps(produce(config, args.scope, args.split, args.benchmark_root,
                                 args.setting, args.datasets, args.nyu_root, args.device)))
    elif args.action == "sync-project":
        remote.sync_project(PROJECT.parent)
    elif args.action == "migrate":
        from explore_a.cache.migration import migrate
        if args.host == "all":
            result = remote.ssh(["bash", f"{remote.REMOTE_PROJECT}/Proj/dgx_cache.sh", "migrate",
                                 "--base", REMOTE_ROOT, "--from-recipe", args.from_recipe,
                                 "--recipe", args.recipe, "--host", "local"])
            print(result)
        print(json.dumps(migrate(base, args.from_recipe, args.recipe), indent=2))
    elif args.action == "plan":
        from explore_a.cache.plan import prepare_plan
        from explore_a.cache.locks import process_lock
        with process_lock(Path(base).expanduser() / "coordinator.lock", "Cache coordinator"):
            prepare_plan(load_config(args.config), args.recipe, base)
    elif args.action == "policy":
        from explore_a.cache.catalog import refresh_catalog
        from explore_a.cache.locks import process_lock
        from explore_a.cache.plan import write_hybrid_config
        from explore_a.cache.policy import refresh_policy
        with process_lock(Path(base).expanduser() / "coordinator.lock", "Cache coordinator"):
            with process_lock(Path(base).expanduser() / "worker.lock", "Local worker"):
                config = load_config(args.config)
                refresh_policy(root, config)
                write_hybrid_config(root, config)
                rows = refresh_catalog(root)
                print(json.dumps({"videos": len(rows), "skipped_length": sum(
                    not row["eligible"] for row in rows),
                    "catalog": str(root / "cache_status.csv")}))
    elif args.action == "pipeline":
        from explore_a.cache.coordinator import run
        run(load_config(args.config), args.recipe, base, workers=args.workers)
    elif args.action == "worker":
        if args.host == "local":
            from explore_a.cache.local_worker import run
        else:
            from explore_a.cache.worker import run
        run(base, args.once)
    elif args.action == "remote":
        from explore_a.cache.worker import control
        print(json.dumps(control(args.operation, base, args.recipe, args.job)))
    else:
        from explore_a.cache.management import retry, selected_hosts, status, stop
        hosts = selected_hosts(root, args.host)
        if args.action == "stop":
            result = stop(root, base, hosts, args.host in (None, "all"))
        elif args.action == "retry":
            result = retry(root, base, args.job, hosts)
        else:
            result = status(root, hosts)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
