from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import pickle
import os
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['THUMB_FOLDER'] = 'static/thumbnails'

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['THUMB_FOLDER'], exist_ok=True)

# Load face database
with open("face_index.pkl", "rb") as f:
    face_db = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded image
        file = request.files['query']
        if not file:
            return "No file uploaded", 400
        filename = secure_filename(file.filename)
        query_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(query_path)

        # Run face recognition
        query_image = face_recognition.load_image_file(query_path)
        query_encodings = face_recognition.face_encodings(query_image)

        results = []
        for query_encoding in query_encodings:
            for entry in face_db:
                match = face_recognition.compare_faces([entry["encoding"]], query_encoding, tolerance=0.6)
                if match[0]:
                    image_path = entry["path"]
                    thumb_name = os.path.basename(image_path)
                    thumb_path = os.path.join(app.config['THUMB_FOLDER'], thumb_name)

                    # Generate thumbnail if missing
                    if not os.path.exists(thumb_path):
                        try:
                            with Image.open(image_path) as img:
                                img.thumbnail((400, 4000), Image.LANCZOS)
                                img.save(thumb_path)
                        except Exception as e:
                            print(f"Error generating thumbnail for {image_path}: {e}")
                            continue

                    results.append({
                        "full": image_path,
                        "thumb": thumb_path
                    })

        return render_template("results.html", results=results, query=filename)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)