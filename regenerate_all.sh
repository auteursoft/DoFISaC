#!/bin/bash
# Regenerate all components for the DoFISaC project

PHOTO_DIR="$1"
if [ -z "$PHOTO_DIR" ]; then
  echo "Usage: $0 <path_to_photos>"
  exit 1
fi

echo "🔁 Running face indexer..."
python3 face-indexer.py "$PHOTO_DIR"

echo "🔁 Running clustering..."
python3 clustering.py

echo "✅ Regeneration complete."
