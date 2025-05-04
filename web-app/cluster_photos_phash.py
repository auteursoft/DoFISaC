import os
from pathlib import Path
from PIL import Image
import imagehash
from sklearn.cluster import DBSCAN
import numpy as np
from shutil import copy2

PHOTO_DIR = "app/static/indexed_faces"
OUTPUT_DIR = "phash_clusters"
HASH_SIZE = 16  # Higher values improve sensitivity (default 8)
HASH_FUNC = imagehash.phash  # Can also try dhash or average_hash

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = list(Path(PHOTO_DIR).glob("*.*"))
hashes = []
valid_paths = []

for path in image_paths:
    try:
        img = Image.open(path).convert("RGB")
        hash_val = HASH_FUNC(img, hash_size=HASH_SIZE)
        hashes.append(hash_val.hash.flatten())  # numpy array
        valid_paths.append(path)
    except Exception as e:
        print(f"Skipping {path.name}: {e}")

hash_array = np.array(hashes)

# Use DBSCAN to cluster by Hamming distance
clustering = DBSCAN(eps=0.25, min_samples=2, metric='hamming').fit(hash_array)

for label in set(clustering.labels_):
    cluster_dir = Path(OUTPUT_DIR) / f"cluster_{label}" if label != -1 else Path(OUTPUT_DIR) / "unclustered"
    cluster_dir.mkdir(parents=True, exist_ok=True)

for path, label in zip(valid_paths, clustering.labels_):
    label_folder = Path(OUTPUT_DIR) / f"cluster_{label}" if label != -1 else Path(OUTPUT_DIR) / "unclustered"
    copy2(path, label_folder / path.name)

print(f"Clustered {len(valid_paths)} images using perceptual hashing into '{OUTPUT_DIR}'")
