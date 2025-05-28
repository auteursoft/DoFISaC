import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import imagehash
from sklearn.cluster import DBSCAN
from shutil import copy2

PHOTO_DIR = "app/static/indexed_faces"
OUTPUT_DIR = "phash_clusters_retrained"
FEEDBACK_FILE = "phash_feedback.json"
HASH_SIZE = 16
HASH_FUNC = imagehash.phash

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = list(Path(PHOTO_DIR).glob("*.*"))
hashes = []
valid_paths = []

for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        hash_val = HASH_FUNC(img, hash_size=HASH_SIZE)
        hashes.append(hash_val.hash.flatten())  # Flattened numpy array
        valid_paths.append(str(path))
    except Exception as e:
        print(f"Skipping {path}: {e}")

hash_array = np.array(hashes)

# Load feedback and influence clustering
feedback = {}
if Path(FEEDBACK_FILE).exists():
    with open(FEEDBACK_FILE, "r") as f:
        feedback = json.load(f)

# Adjust clustering by assigning must-link and cannot-link constraints
# Simplified: reduce eps for images marked as incorrect, cluster tighter
eps_map = {}

for cluster, images in feedback.items():
    for img_path, label in images.items():
        eps_map[img_path] = 0.15 if label == "incorrect" else 0.25

eps_values = [eps_map.get(p, 0.25) for p in valid_paths]
mean_eps = sum(eps_values) / len(eps_values)

clustering = DBSCAN(eps=mean_eps, min_samples=2, metric='hamming').fit(hash_array)

for label in set(clustering.labels_):
    label_dir = Path(OUTPUT_DIR) / f"cluster_{label}" if label != -1 else Path(OUTPUT_DIR) / "unclustered"
    label_dir.mkdir(parents=True, exist_ok=True)

for path, label in zip(valid_paths, clustering.labels_):
    dest_folder = Path(OUTPUT_DIR) / f"cluster_{label}" if label != -1 else Path(OUTPUT_DIR) / "unclustered"
    copy2(path, dest_folder / Path(path).name)

print(f"Retrained clustering based on feedback into '{OUTPUT_DIR}'")
