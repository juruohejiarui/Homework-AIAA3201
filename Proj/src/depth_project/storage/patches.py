"""Apply only the versioned patches to official submodules."""
import subprocess

from depth_project.paths import PROJECT, THIRD_PARTY

PATCHES = [
    ("depth-pro-python313-numpy.patch", "ml-depth-pro"),
    ("vda-benchmark-compat.patch", "Video-Depth-Anything"),
    ("vda-preprocessing-corrections.patch", "Video-Depth-Anything"),
]


def main():
    for name, repository in PATCHES:
        patch = PROJECT / "patches" / name
        command = ["git", "-C", str(THIRD_PARTY / repository), "apply"]
        applied = subprocess.run(command + ["--reverse", "--check", str(patch)], capture_output=True)
        if applied.returncode == 0:
            print(f"Already applied: {name}")
            continue
        subprocess.run(command + ["--check", str(patch)], check=True)
        subprocess.run(command + [str(patch)], check=True)
        print(f"Applied: {name}")


if __name__ == "__main__":
    main()
