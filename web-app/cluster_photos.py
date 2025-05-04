import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import copy2

PHOTO_DIR = "app/static/indexed_faces"
OUTPUT_DIR = "clusters"
N_CLUSTERS = 5  # Adjust depending on the diversity of backgrounds

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_background_feature(image_path, resize_dim=(100, 100)):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    image = cv2.resize(image, resize_dim)
    return image.mean(axis=(0, 1))  # Mean color as a basic background descriptor

# Load all features
features = []
image_paths = []

for path in Path(PHOTO_DIR).glob("*.*"):
    feat = extract_background_feature(path)
    if feat is not None:
        features.append(feat)
        image_paths.append(path)

features = np.array(features)

# Cluster based on background features
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(features)

# Copy images into cluster subfolders
for cluster_id in range(N_CLUSTERS):
    cluster_folder = Path(OUTPUT_DIR) / f"cluster_{cluster_id}"
    cluster_folder.mkdir(parents=True, exist_ok=True)

for path, label in zip(image_paths, labels):
    copy2(path, Path(OUTPUT_DIR) / f"cluster_{label}" / path.name)

print(f"Clustered {len(image_paths)} images into {N_CLUSTERS} folders in '{OUTPUT_DIR}'")
