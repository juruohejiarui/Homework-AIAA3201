#!/usr/bin/env bash
# Thin entry: the user explicitly starts each evaluation scope.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m explore_a.evaluation.comparison "$@"
