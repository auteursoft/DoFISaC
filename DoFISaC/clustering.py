import pickle
import numpy as np
from sklearn.cluster import DBSCAN
import os
import json
from pathlib import Path
import multiprocessing

PKL_PATH = "face_index.pkl"
PHASH_OUTPUT_JSON = "phash_clusters.json"
BG_OUTPUT_JSON = "bg_clusters.json"

def load_data():
    with open(PKL_PATH, "rb") as f:
        face_db = pickle.load(f)

    vectors = []
    entries = []
    for entry in face_db:
        if entry["face_vec"] is not None and entry["bg_vec"] is not None:
            combined = np.concatenate([entry["face_vec"], entry["bg_vec"]])
            vectors.append(combined)
            entries.append(entry)
    return np.array(vectors).astype("float32"), entries

def cluster_vectors(vectors):
    clustering = DBSCAN(eps=30, min_samples=3, metric='euclidean').fit(vectors)
    return clustering.labels_

def build_cluster_json(labels, entries):
    cluster_map = {}
    for label, entry in zip(labels, entries):
        cluster = f"cluster_{label}" if label >= 0 else "noise"
        cluster_map.setdefault(cluster, []).append({
            "thumb": entry["thumb_name"],
            "path": entry["path"]
        })
    return cluster_map

def main():
    print("🔍 Loading data...")
    vectors, entries = load_data()

    print(f"🤖 Clustering {len(vectors)} images...")
    labels = cluster_vectors(vectors)

    print("🧩 Building JSON cluster map...")
    cluster_map = build_cluster_json(labels, entries)

    print("💾 Saving cluster files...")
    with open(PHASH_OUTPUT_JSON, "w") as f:
        json.dump(cluster_map, f, indent=2)
    with open(BG_OUTPUT_JSON, "w") as f:
        json.dump(cluster_map, f, indent=2)

    print("✅ Clustering complete and JSONs written.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()