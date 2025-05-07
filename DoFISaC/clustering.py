
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import os
import shutil
import multiprocessing

CLUSTER_OUTPUT_DIR = "static/clusters"
PKL_PATH = "face_index.pkl"

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

def save_cluster(label_entry):
    label, entry = label_entry
    cluster_name = f"cluster_{label}" if label >= 0 else "noise"
    cluster_path = Path(CLUSTER_OUTPUT_DIR) / cluster_name
    cluster_path.mkdir(parents=True, exist_ok=True)
    dest = cluster_path / Path(entry["path"]).name
    try:
        shutil.copy(entry["path"], dest)
    except Exception as e:
        print(f"Failed to copy {entry['path']}: {e}")

def main():
    if os.path.exists(CLUSTER_OUTPUT_DIR):
        shutil.rmtree(CLUSTER_OUTPUT_DIR)
    os.makedirs(CLUSTER_OUTPUT_DIR, exist_ok=True)

    print("🔍 Loading face and background vectors...")
    vectors, valid_entries = load_vectors()

    print(f"🤖 Clustering {len(vectors)} images...")
    clustering = DBSCAN(eps=30, min_samples=3, metric='euclidean').fit(vectors)
    labels = clustering.labels_

    print("💾 Saving clustered images...")
    with multiprocessing.Pool() as pool:
        pool.map(save_cluster, zip(labels, valid_entries))

    print("✅ Clustering complete.")

if __name__ == "__main__":
    main()
