#!/bin/bash

echo "📦 Installed packages (no versions):"
cat /app/packages.txt

# Copy package list to output
cp /app/packages.txt /output/packages.txt

# Check config file
if [[ ! -f /config/folders.txt ]]; then
  echo "❌ Missing config file: /config/folders.txt"
  exit 1
fi

echo "📁 Reading folder list from /config/folders.txt"
while IFS= read -r folder || [[ -n "$folder" ]]; do
  folder_trimmed=$(echo "$folder" | xargs)  # trim whitespace
  if [[ -d "$folder_trimmed" ]]; then
    echo "🔍 Processing: $folder_trimmed"
    python index-and-cluster.py "$folder_trimmed"
  else
    echo "⚠️ Skipping non-directory: $folder_trimmed"
  fi
done < /config/folders.txt