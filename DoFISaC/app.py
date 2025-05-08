# app.py (regenerated and corrected)
from flask import Flask, render_template, request, jsonify, url_for
import os, json, pickle
from pathlib import Path
import numpy as np
from insightface.app import FaceAnalysis
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
THUMBNAIL_DIR = "static/thumbnails"
INDEX_PATH = "face_index.pkl"
FEEDBACK_PATH = "static/feedback.json"
CLUSTER_FEEDBACK_PATH = "static/cluster_feedback.json"
PHASH_CLUSTER_PATH = "phash_clusters.json"
BG_CLUSTER_PATH = "bg_clusters.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

with open(INDEX_PATH, "rb") as f:
    face_db = pickle.load(f)

face_vectors = np.array([e["face_vec"] for e in face_db if e["face_vec"] is not None], dtype="float32")
bg_vectors = np.array([e["bg_vec"] for e in face_db if e["bg_vec"] is not None], dtype="float32")
face_index = faiss.IndexFlatL2(face_vectors.shape[1])
bg_index = faiss.IndexFlatL2(bg_vectors.shape[1])
face_index.add(face_vectors)
bg_index.add(bg_vectors)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        f = request.files["file"]
        save_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(save_path)
        img = Image.open(save_path).convert("RGB")
        img_np = np.array(img)
        face_vec = face_model.get(img_np)[0].embedding if face_model.get(img_np) else None
        inputs = clip_processor(images=img, return_tensors="pt", padding=True)
        bg_vec = clip_model.get_image_features(**inputs)[0].detach().numpy()

        results = []
        if face_vec is not None:
            dists, idxs = face_index.search(np.array([face_vec], dtype="float32"), 50)
            for dist, idx in zip(dists[0], idxs[0]):
                results.append((face_db[idx], dist))
        if bg_vec is not None:
            dists, idxs = bg_index.search(np.array([bg_vec], dtype="float32"), 50)
            for dist, idx in zip(dists[0], idxs[0]):
                results.append((face_db[idx], dist))

        # Deduplicate and annotate
        seen, unique = set(), []
        for entry, dist in results:
            if entry["path"] not in seen:
                seen.add(entry["path"])
                entry["match"] = "yes"
                entry["distance"] = f"{dist:.4f}"
                unique.append(entry)

        page_count = 1  # for template compat
        return render_template("search_results.html", results=unique, query=f.filename, page_count=page_count)

    return render_template("search.html", page_count=1)

@app.route("/clusters/phash")
def clusters_phash():
    with open(PHASH_CLUSTER_PATH) as f:
        clusters = json.load(f)
    return render_template("clusters_phash.html", clusters=clusters)

@app.route("/clusters/bg")
def clusters_bg():
    with open(BG_CLUSTER_PATH) as f:
        clusters = json.load(f)
    return render_template("clusters_bg.html", clusters=clusters)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    path, label, context = data["image"], data["label"], data["context"]
    feedback_path = FEEDBACK_PATH if context == "face_search" else CLUSTER_FEEDBACK_PATH
    feedback = {}
    if os.path.exists(feedback_path):
        with open(feedback_path, "r") as f:
            feedback = json.load(f)
    feedback[path] = label
    with open(feedback_path, "w") as f:
        json.dump(feedback, f, indent=2)
    return jsonify(status="ok")

@app.route("/retrain")
def retrain():
    return "Retraining not implemented yet.", 501

if __name__ == "__main__":
    app.run(debug=True)