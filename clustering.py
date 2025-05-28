import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import os
import shutil

CLUSTER_OUTPUT_DIR = "static/clusters"
PKL_PATH = "face_index.pkl"

with open(PKL_PATH, "rb") as f:
    face_db = pickle.load(f)

# Combine all features for clustering
vectors = []
valid_entries = []
for entry in face_db:
    if entry["face_vec"] is not None and entry["bg_vec"] is not None:
        combined = np.concatenate([entry["face_vec"], entry["bg_vec"]])
        vectors.append(combined)
        valid_entries.append(entry)

vectors = np.array(vectors).astype("float32")

# Use DBSCAN with automatic epsilon guess (can be tuned)
print("Clustering", len(vectors), "images...")
clustering = DBSCAN(eps=30, min_samples=3, metric='euclidean').fit(vectors)
labels = clustering.labels_

# Save clusters to folders
output_dir = Path(CLUSTER_OUTPUT_DIR)
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

for label, entry in zip(labels, valid_entries):
    cluster_name = f"cluster_{label}" if label >= 0 else "noise"
    cluster_path = output_dir / cluster_name
    cluster_path.mkdir(parents=True, exist_ok=True)
    dest = cluster_path / Path(entry["path"]).name
    try:
        shutil.copy(entry["path"], dest)
    except Exception as e:
        print(f"Failed to copy {entry['path']}: {e}")

print("✅ Clustering complete.")
