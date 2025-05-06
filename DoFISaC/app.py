from flask import Flask, render_template, request, jsonify, url_for
import os
import json
import pickle
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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        seen = set()
        unique = []
        for r in results:
            if r["path"] not in seen:
                unique.append(r)
                seen.add(r["path"])
        return render_template("search_results.html", results=unique, query=f.filename)
    return render_template("search.html")

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

if __name__ == "__main__":
    app.run(debug=True)