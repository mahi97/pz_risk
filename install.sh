#!/usr/bin/env bash
# Bootstrap a complete pz-risk development environment with uv.
# Installs the package, runtime deps (including evosax / JAX / SB3 / PettingZoo),
# and the dev/test extra. Optionally runs the test suite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

RUN_TESTS=0
for arg in "$@"; do
    case "${arg}" in
        --test|-t)
            RUN_TESTS=1
            ;;
        --help|-h)
            cat <<'EOF'
Usage: ./install.sh [--test]

  Installs uv if needed, creates the project virtualenv, and syncs all
  dependencies declared in pyproject.toml (runtime + future RL/evo + tests).

  --test, -t   Run pytest with coverage after the install.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            echo "Usage: ./install.sh [--test]" >&2
            exit 2
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing via https://astral.sh/uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "Using uv $(uv --version)"

# Prefer the project's declared Python; uv will download a compatible one if needed.
uv python install 3.12

echo "Syncing project, runtime, RL/evolutionary, and test dependencies..."
# CPU wheels by default (see pyproject.toml). Use UV_INDEX / extras later for CUDA.
uv sync --all-groups

echo "Verifying the install..."
uv run python - <<'PY'
import importlib

modules = [
    "pz_risk",
    "pz_risk.core.board",
    "pz_risk.envs",
    "gymnasium",
    "pettingzoo",
    "networkx",
    "numpy",
    "scipy",
    "matplotlib",
    "loguru",
    "torch",
    "stable_baselines3",
    "evosax",
    "jax",
    "flax",
    "optax",
    "chex",
    "gymnax",
    "supersuit",
    "pytest",
]
failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report every failure
        failed.append(f"{name}: {exc}")

from pz_risk import __version__, make
from pz_risk.core.board import BOARDS

env = make("Risk-Normal-2-v0")
env.reset()
env.close()

print(f"pz-risk {__version__}")
print("maps:", sorted(BOARDS))
if failed:
    print("WARNING: some optional imports failed:")
    for item in failed:
        print(" -", item)
    raise SystemExit(1)
print("install ok")
PY

if [[ "${RUN_TESTS}" -eq 1 ]]; then
    echo "Running pytest with coverage..."
    uv run pytest
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Or run tools via:     uv run pytest / uv run python"
echo "Manual play:          uv run pz-risk-manual --help"
echo "Benchmark:            uv run pz-risk-benchmark --help"
