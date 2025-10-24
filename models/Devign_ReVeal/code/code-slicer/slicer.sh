#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 3 ]]; then
  inpdir=$1
  filename=$2
  outdir=$3

  script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  workdir="$(mktemp -d)"
  trap 'rm -rf "$workdir"' EXIT

  mkdir -p "$workdir/tmp" "$outdir"
  cp "$inpdir/$filename" "$workdir/tmp/$filename"

  ( cd "$workdir" && "$script_dir/joern/joern-parse" tmp/ )

  if compgen -G "$workdir/parsed/*" > /dev/null; then
    cp -r "$workdir/parsed/"* "$outdir/"
  fi
else
  echo 'Wrong Argument!.'
  echo 'Usage: slicer.sh <Directory of the C File> <Name of the C File> <Output Directory>'
fi
