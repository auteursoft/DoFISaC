from flask import Blueprint, render_template, request, redirect, url_for
import face_recognition
import pickle
import os
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

# Load face index once at startup
with open("face_index.pkl", "rb") as f:
    face_db = pickle.load(f)

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return redirect(url_for('main.index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('main.index'))

    filename = secure_filename(file.filename)
    upload_path = os.path.join('app/static/uploads', filename)
    file.save(upload_path)

    query_image = face_recognition.load_image_file(upload_path)
    query_encodings = face_recognition.face_encodings(query_image)

    matches = set()
    for query_encoding in query_encodings:
        for entry in face_db:
            match = face_recognition.compare_faces([entry["encoding"]], query_encoding, tolerance=0.6)
            if match[0]:
                matches.add(entry["path"])

    return render_template('results.html', matches=matches, uploaded=upload_path)
