import os
from pathlib import Path
from PIL import Image
import face_recognition
import pickle
import numpy as np
import imagehash
from shutil import copy2
import cv2
from sklearn.cluster import DBSCAN, KMeans

PHOTO_DIR = "/Volumes/super_54/google/sean.goggins/Google Photos" 
THUMB_DIR = "static/thumbnails"
CLUSTER_PHASH_DIR = "static/clusters/phash"
CLUSTER_BG_DIR = "static/clusters/bg"
INDEX_FILE = "face_index.pkl"
HASH_SIZE = 16
THUMB_WIDTH = 400
N_BG_CLUSTERS = 5

os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(CLUSTER_PHASH_DIR, exist_ok=True)
os.makedirs(CLUSTER_BG_DIR, exist_ok=True)

face_db = []
hashes = []
hash_paths = []
bg_features = []
bg_paths = []

# === Index and Thumbnail ===
for path in Path(PHOTO_DIR).glob("*.*"):
    try:
        img = Image.open(path).convert("RGB")
        rel_path = os.path.relpath(path)
        # === Face Encoding ===
        encodings = face_recognition.face_encodings(np.array(img))
        for encoding in encodings:
            face_db.append({"path": rel_path, "encoding": encoding})

        # === Thumbnail ===
        thumb_path = Path(THUMB_DIR) / path.name
        if not thumb_path.exists():
            img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 10000), Image.LANCZOS)
            img.save(thumb_path)

        # === Perceptual Hash Clustering Prep ===
        hash_val = imagehash.phash(img, hash_size=HASH_SIZE)
        hashes.append(hash_val.hash.flatten())
        hash_paths.append(path)

        # === Background Clustering Prep ===
        cv_img = cv2.imread(str(path))
        if cv_img is not None:
            resized = cv2.resize(cv_img, (100, 100))
            bg_features.append(resized.mean(axis=(0, 1)))
            bg_paths.append(path)

    except Exception as e:
        print(f"❌ Skipping {path}: {e}")

# === Save Face Index ===
with open(INDEX_FILE, "wb") as f:
    pickle.dump(face_db, f)
print(f"✅ Saved face index to {INDEX_FILE}")

# === Cluster by Perceptual Hash ===
hash_array = np.array(hashes)
phash_labels = DBSCAN(eps=0.25, min_samples=2, metric='hamming').fit_predict(hash_array)

for path, label in zip(hash_paths, phash_labels):
    folder = Path(CLUSTER_PHASH_DIR) / f"cluster_{label if label != -1 else 'unclustered'}"
    folder.mkdir(parents=True, exist_ok=True)
    copy2(path, folder / path.name)

# === Cluster by Background Color ===
bg_features = np.array(bg_features)
bg_labels = KMeans(n_clusters=N_BG_CLUSTERS, random_state=42).fit_predict(bg_features)

for path, label in zip(bg_paths, bg_labels):
    folder = Path(CLUSTER_BG_DIR) / f"cluster_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    copy2(path, folder / path.name)

print("✅ Clustering complete.")