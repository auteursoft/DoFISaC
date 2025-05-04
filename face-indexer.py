import os
import face_recognition
from tqdm import tqdm
import pickle

image_dir = "/path/to/all/photos"
face_db = []

for root, _, files in os.walk(image_dir):
    for file in tqdm(files):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(root, file)
            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                for encoding in encodings:
                    face_db.append({
                        "path": filepath,
                        "encoding": encoding
                    })
            except Exception as e:
                print(f"⚠️ Skipped {file}: {e}")

with open("face_index.pkl", "wb") as f:
    pickle.dump(face_db, f)