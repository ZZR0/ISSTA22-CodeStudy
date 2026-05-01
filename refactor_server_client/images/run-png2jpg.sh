#!/bin/bash

for f in ./*.png; do
  [ -e "$f" ] || continue
  sips -s format jpeg "$f" --out "${f%.png}.jpg" >/dev/null
  echo "Converted: $f -> ${f%.png}.jpg"
done
