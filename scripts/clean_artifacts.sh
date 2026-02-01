#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifacts_dir="$root_dir/artifacts"

if [[ ! -d "$artifacts_dir" ]]; then
  exit 0
fi

find "$artifacts_dir" -type f ! -name ".gitkeep" -print -delete
find "$artifacts_dir" -type d -empty -print -delete
