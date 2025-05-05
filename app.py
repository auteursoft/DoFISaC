from flask import Flask, render_template, request, redirect, url_for
import os
import json
import pickle
from pathlib import Path
import face_recognition
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
CLUSTER_PHASH_DIR = "static/clusters/phash"
CLUSTER_BG_DIR = "static/clusters/bg"
FACE_INDEX_PATH = "face_index.pkl"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def load_index(path=FACE_INDEX_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

def search_faces(image_path, index, tolerance=0.6):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return []
    matches = []
    for encoding in encodings:
        for entry in index:
            if face_recognition.compare_faces([entry["encoding"]], encoding, tolerance=tolerance)[0]:
                matches.append(entry["path"])
    return matches

@app.route("/")
def index():
    return render_template("index.html")

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

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            index = load_index()
            results = search_faces(filepath, index)
            return render_template("search_results.html", results=results, query=filename)
    return render_template("search.html")

@app.route("/feedback", methods=["POST"])
def feedback():
    fb_file = "static/feedback.json"
    feedback = {}
    if os.path.exists(fb_file):
        with open(fb_file, "r") as f:
            feedback = json.load(f)
    data = request.get_json()
    image = data["image"]
    label = data["label"]
    context = data["context"]
    if context not in feedback:
        feedback[context] = {}
    feedback[context][image] = label
    with open(fb_file, "w") as f:
        json.dump(feedback, f, indent=2)
    return "ok"

if __name__ == "__main__":
    app.run(debug=True)
