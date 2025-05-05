import os
import argparse
from pathlib import Path
from PIL import Image
import face_recognition
import pickle
import numpy as np
import imagehash
from shutil import copy2
import cv2
from sklearn.cluster import DBSCAN, KMeans
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP.*")

THUMB_DIR = "static/thumbnails"
CLUSTER_PHASH_DIR = "static/clusters/phash"
CLUSTER_BG_DIR = "static/clusters/bg"
INDEX_FILE = "face_index.pkl"
HASH_SIZE = 16
THUMB_WIDTH = 400
N_BG_CLUSTERS = 5
MAX_DEPTH = 3

os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(CLUSTER_PHASH_DIR, exist_ok=True)
os.makedirs(CLUSTER_BG_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Index and cluster images.")
parser.add_argument("paths", nargs="*", help="Image root paths or a file containing newline-separated image paths.")
parser.add_argument("--thumbnails-only", action="store_true", help="Only regenerate thumbnails.")
parser.add_argument("--cluster-only", action="store_true", help="Only run clustering on existing images.")
args = parser.parse_args()

def is_valid_image_file(path):
    return path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]

def collect_image_paths(sources, max_depth=3):
    collected = []
    for source in sources:
        p = Path(source)
        if p.is_file() and p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            with open(p) as f:
                for line in f:
                    path = Path(line.strip())
                    if path.exists() and is_valid_image_file(path):
                        collected.append(path)
        elif p.is_dir():
            base_depth = len(p.resolve().parts)
            for fp in p.rglob("*"):
                if is_valid_image_file(fp) and len(fp.resolve().parts) - base_depth <= max_depth:
                    collected.append(fp)
        elif p.is_file() and is_valid_image_file(p):
            collected.append(p)
    return collected

image_paths = collect_image_paths(args.paths, MAX_DEPTH)

def process_image(path):
    result = {
        "path": str(path),
        "rel_path": os.path.relpath(path),
        "face_encodings": [],
        "thumb_path": None,
        "phash": None,
        "bg_feat": None
    }

    try:
        img = Image.open(path).convert("RGB")
        np_img = np.array(img)

        if not args.cluster_only:
            encodings = face_recognition.face_encodings(np_img)
            result["face_encodings"] = encodings

        thumb_path = Path(THUMB_DIR) / path.name
        if not thumb_path.exists():
            img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 10000), Image.LANCZOS)
            img.save(thumb_path)
        result["thumb_path"] = str(thumb_path)

        if not args.thumbnails_only:
            result["phash"] = imagehash.phash(img, hash_size=HASH_SIZE).hash.flatten()
            cv_img = cv2.imread(str(path))
            if cv_img is not None:
                resized = cv2.resize(cv_img, (100, 100))
                result["bg_feat"] = resized.mean(axis=(0, 1))

    except Exception as e:
        print(f"❌ Error processing {path}: {e}")

    return result

def main():
    print(f"🧠 Processing {len(image_paths)} images with {cpu_count()} cores...")
    with Pool() as pool:
        processed = list(pool.map(process_image, image_paths))

    face_db = []
    phash_vectors = []
    phash_paths = []
    bg_features = []
    bg_paths = []

    for item in processed:
        for encoding in item["face_encodings"]:
            face_db.append({"path": item["rel_path"], "encoding": encoding})
        if item["phash"] is not None:
            phash_vectors.append(item["phash"])
            phash_paths.append(Path(item["path"]))
        if item["bg_feat"] is not None:
            bg_features.append(item["bg_feat"])
            bg_paths.append(Path(item["path"]))

    if not args.cluster_only:
        with open(INDEX_FILE, "wb") as f:
            pickle.dump(face_db, f)
        print(f"✅ Saved face index to {INDEX_FILE} ({len(face_db)} face encodings)")

    if not args.thumbnails_only:
        print("🔍 Clustering by perceptual hash...")
        if phash_vectors:
            phash_array = np.array(phash_vectors)
            phash_labels = DBSCAN(eps=0.25, min_samples=2, metric='hamming').fit_predict(phash_array)
            for path, label in zip(phash_paths, phash_labels):
                folder = Path(CLUSTER_PHASH_DIR) / f"cluster_{label if label != -1 else 'unclustered'}"
                folder.mkdir(parents=True, exist_ok=True)
                copy2(path, folder / path.name)
            print(f"✅ Clustered {len(phash_paths)} images by perceptual hash.")
        else:
            print("⚠️ No valid images found for phash clustering.")

        print("🔍 Clustering by background features...")
        if len(bg_features) > 1:
            actual_clusters = min(N_BG_CLUSTERS, len(bg_features))
            bg_labels = KMeans(n_clusters=actual_clusters, random_state=42).fit_predict(np.array(bg_features))
            for path, label in zip(bg_paths, bg_labels):
                folder = Path(CLUSTER_BG_DIR) / f"cluster_{label}"
                folder.mkdir(parents=True, exist_ok=True)
                copy2(path, folder / path.name)
            print(f"✅ Clustered {len(bg_paths)} images by background ({actual_clusters} clusters).")
        else:
            print("⚠️ Not enough images for background clustering.")

    print("🏁 Done.")

if __name__ == "__main__":
    main()
