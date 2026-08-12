#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_bin="${project_dir}/.venv/bin"

"${venv_bin}/ruff" check "${project_dir}"
"${venv_bin}/ruff" format --check "${project_dir}"
"${venv_bin}/pytest" --cov=omnitop --cov-report=term-missing "${project_dir}/tests"
