#!/usr/bin/env bash
set -euo pipefail

dir=$1                     # e.g. ./data/labels_41375
chunk=${2:-500}            # optional: files per progress batch
jobs=${PARALLEL:-$(nproc)} # override with PARALLEL=8 bash parse_all.sh ...

mkdir -p "$dir/parsed"

total=$(ls "$dir/raw_code" | wc -l)
offset=0

while [ "$offset" -lt "$total" ]; do
  echo "Processing files $offset – $((offset + chunk - 1)) / $total..."

  ls "$dir/raw_code" \
  | awk -v o="$offset" -v c="$chunk" 'NR>o && NR<=o+c' \
  | xargs -I{} -n1 -P"$jobs" \
      bash "$(dirname "$0")/slicer.sh" "$dir/raw_code" "{}" "$dir/parsed"

  offset=$((offset + chunk))
done
