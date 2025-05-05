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