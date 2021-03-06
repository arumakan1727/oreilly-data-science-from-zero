#!/usr/bin/env bash
set -eu

doctest_target_dir=$1

for f in $(find "$doctest_target_dir" -type f -name '*.py'); do
  python -m doctest $f
  echo "doctest: [OK] $f"
done
