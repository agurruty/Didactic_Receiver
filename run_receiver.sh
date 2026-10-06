#!/usr/bin/env bash
# Example: PYTHON=/usr/bin/python3 ./run_receiver.sh recordings --output runs/demo --top 8
set -euo pipefail
receiver_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$receiver_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -c 'from receiver.common import launch; from receiver.pipeline import main; launch(main)' "$@"
