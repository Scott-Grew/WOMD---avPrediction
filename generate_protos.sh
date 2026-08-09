#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
protoc --proto_path=proto --python_out=. \
  proto/waymo_open_dataset/protos/scenario.proto \
  proto/waymo_open_dataset/protos/map.proto
touch waymo_open_dataset/__init__.py waymo_open_dataset/protos/__init__.py
echo "protos generated"
