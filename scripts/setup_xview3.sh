#!/usr/bin/env bash
# Fetch the public assets the xView3 detector needs (code + pretrained weights).
# Idempotent. Scene imagery + labels are credentialed (iuu.xview.us) and not fetched here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI2_DIR="$ROOT/third_party/sar_vessel_detect"
WEIGHTS="$ROOT/data/weights/xview3-combo8.pth"
WEIGHTS_URL="https://ai2-prior-sarfish.s3.us-west-2.amazonaws.com/public/sarfish-models/xview3-nov20-combo8/model.pth"

# 1) AI2 sar_vessel_detect source (Apache-2.0) — the model architecture + inference code.
if [ -d "$AI2_DIR/src/xview3" ]; then
  echo "AI2 source present: $AI2_DIR"
else
  echo "Cloning AI2 sar_vessel_detect..."
  git clone --depth 1 https://github.com/allenai/sar_vessel_detect "$AI2_DIR"
fi

# 2) Pretrained weights (public S3, ~210 MB).
if [ -f "$WEIGHTS" ]; then
  echo "Weights present: $WEIGHTS"
else
  echo "Downloading weights..."
  mkdir -p "$(dirname "$WEIGHTS")"
  curl -fSL -o "$WEIGHTS" "$WEIGHTS_URL"
fi

echo "Done. Scene imagery + labels are credentialed: register at https://iuu.xview.us"
echo "and place a scene under data/xview3/scenes/<scene_id>/ and labels at data/xview3/validation.csv"
