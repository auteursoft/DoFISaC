import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import os
import json

CLUSTER_OUTPUT_JSON_PHASH = "phash_clusters.json"
CLUSTER_OUTPUT_JSON_BG = "bg_clusters.json"
PKL_PATH = "face_index.pkl"
THUMBNAIL_DIR = "static/thumbnails"

def load_vectors():
    with open(PKL_PATH, "rb") as f:
        face_db = pickle.load(f)

    vectors = []
    valid_entries = []
    for entry in face_db:
        if entry["face_vec"] is not None and entry["bg_vec"] is not None:
            combined = np.concatenate([entry["face_vec"], entry["bg_vec"]])
            vectors.append(combined)
            valid_entries.append(entry)
    return np.array(vectors).astype("float32"), valid_entries

def main():
    print("🔍 Loading vectors...")
    vectors, valid_entries = load_vectors()

    print(f"🤖 Clustering {len(vectors)} images...")
    clustering = DBSCAN(eps=30, min_samples=3, metric='euclidean').fit(vectors)
    labels = clustering.labels_

    cluster_map = {}
    for label, entry in zip(labels, valid_entries):
        cluster_name = f"cluster_{label}" if label >= 0 else "noise"
        cluster_map.setdefault(cluster_name, []).append({
            "thumb": entry["thumb_name"],
            "path": entry["path"]
        })

    with open(CLUSTER_OUTPUT_JSON_PHASH, "w") as f:
        json.dump(cluster_map, f, indent=2)

    with open(CLUSTER_OUTPUT_JSON_BG, "w") as f:
        json.dump(cluster_map, f, indent=2)

    print("✅ Clustering complete and metadata saved.")

if __name__ == "__main__":
    main()