from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import faiss
from insightface.app import FaceAnalysis
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
THUMBNAIL_DIR = "static/thumbnails"
CLUSTER_PHASH_DIR = "static/clusters/phash"
CLUSTER_BG_DIR = "static/clusters/bg"
INDEX_PATH = "face_index.pkl"
FEEDBACK_PATH = "static/feedback.json"
CLUSTER_FEEDBACK_PATH = "static/cluster_feedback.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load index
with open(INDEX_PATH, "rb") as f:
    face_db = pickle.load(f)

face_vectors = np.array([entry["face_vec"] for entry in face_db if entry["face_vec"] is not None]).astype("float32")
bg_vectors = np.array([entry["bg_vec"] for entry in face_db if entry["bg_vec"] is not None]).astype("float32")
face_index = faiss.IndexFlatL2(face_vectors.shape[1])
bg_index = faiss.IndexFlatL2(bg_vectors.shape[1])
face_index.add(face_vectors)
bg_index.add(bg_vectors)

def extract_face_embedding(image_np):
    faces = face_model.get(image_np)
    return faces[0].embedding if faces else None

def extract_clip_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    return clip_model.get_image_features(**inputs)[0].detach().numpy()

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
        face_vec = extract_face_embedding(img_np)
        bg_vec = extract_clip_embedding(img)

        results = []
        if face_vec is not None:
            dists, idxs = face_index.search(np.array([face_vec]).astype("float32"), 50)
            for i in idxs[0]:
                results.append(face_db[i])
        if bg_vec is not None:
            dists, idxs = bg_index.search(np.array([bg_vec]).astype("float32"), 50)
            for i in idxs[0]:
                results.append(face_db[i])

        # Remove duplicates
        seen = set()
        unique = []
        for r in results:
            if r["path"] not in seen:
                unique.append(r)
                seen.add(r["path"])
        return render_template("search_results.html", results=unique, query=f.filename)
    return render_template("search.html")

@app.route("/clusters/phash")
def clusters_phash():
    clusters = {}
    base = Path(CLUSTER_PHASH_DIR)
    for cluster_dir in base.glob("cluster_*"):
        clusters[cluster_dir.name] = sorted([f"/{cluster_dir / img.name}" for img in cluster_dir.glob("*")])
    return render_template("clusters_phash.html", clusters=clusters)

@app.route("/clusters/bg")
def clusters_bg():
    clusters = {}
    base = Path(CLUSTER_BG_DIR)
    for cluster_dir in base.glob("cluster_*"):
        clusters[cluster_dir.name] = sorted([f"/{cluster_dir / img.name}" for img in cluster_dir.glob("*")])
    return render_template("clusters_bg.html", clusters=clusters)

@app.route("/retrain")
def retrain():
    return render_template("retrain.html")

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    path = data["image"]
    label = data["label"]
    context = data["context"]
    feedback_path = FEEDBACK_PATH if context == "face_search" else CLUSTER_FEEDBACK_PATH
    feedback = {}
    if os.path.exists(feedback_path):
        with open(feedback_path, "r") as f:
            feedback = json.load(f)
    feedback[path] = label
    with open(feedback_path, "w") as f:
        json.dump(feedback, f, indent=2)
    return jsonify(status="ok")

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)