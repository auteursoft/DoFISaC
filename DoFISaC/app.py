# app.py (patched for correct search result logic)
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

face_vectors = np.array([e["face_vec"] for e in face_db if e["face_vec"] is not None]).astype("float32")
bg_vectors = np.array([e["bg_vec"] for e in face_db if e["bg_vec"] is not None]).astype("float32")
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
    page = int(request.args.get("page", 1))
    per_page = 20
    results = []
    query = None

    if request.method == "POST":
        f = request.files["file"]
        query = f.filename
        save_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(save_path)

        img = Image.open(save_path).convert("RGB")
        img_np = np.array(img)
        face_vec = extract_face_embedding(img_np)
        bg_vec = extract_clip_embedding(img)

        seen = set()
        combined_results = []

        if face_vec is not None:
            dists, idxs = face_index.search(np.array([face_vec]), 100)
            for d, i in zip(dists[0], idxs[0]):
                entry = face_db[i]
                if entry["path"] not in seen:
                    entry.update({"match": "face", "distance": float(d)})
                    combined_results.append(entry)
                    seen.add(entry["path"])

        if bg_vec is not None:
            dists, idxs = bg_index.search(np.array([bg_vec]), 100)
            for d, i in zip(dists[0], idxs[0]):
                entry = face_db[i]
                if entry["path"] not in seen:
                    entry.update({"match": "background", "distance": float(d)})
                    combined_results.append(entry)
                    seen.add(entry["path"])

        results = combined_results

    total = len(results)
    page_count = (total + per_page - 1) // per_page
    paginated_results = results[(page - 1) * per_page : page * per_page]

    return render_template(
        "search.html",
        results=paginated_results,
        query=query,
        page=page,
        page_count=page_count,
    )


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