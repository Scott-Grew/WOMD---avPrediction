#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d waymo_open_dataset ]; then
  ./generate_protos.sh
fi

python3 -m pytest tests/ -q
