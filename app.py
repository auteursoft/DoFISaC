from flask import Flask, render_template, request
import os
from pathlib import Path
import json

app = Flask(__name__)
CLUSTER_PHASH_DIR = "static/clusters/phash"
CLUSTER_BG_DIR = "static/clusters/bg"

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
