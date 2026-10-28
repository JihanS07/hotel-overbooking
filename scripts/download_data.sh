#!/usr/bin/env bash
set -euo pipefail

DEST="data"
URL="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
SHA256="7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"

mkdir -p "$DEST"
curl -fSL --retry 5 -o "$DEST/hotels.csv" "$URL"
echo "$SHA256  $DEST/hotels.csv" | sha256sum -c -
echo "Saved $DEST/hotels.csv"
