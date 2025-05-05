import face_recognition
import os
import pickle
from PIL import Image

# Settings
IMAGE_DIR = "photos"  # folder containing your images
OUTPUT_INDEX = "face_index.pkl"
THUMBNAIL_DIR = "static/thumbnails"
THUMBNAIL_WIDTH = 400

# Ensure thumbnail directory exists
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

face_db = []

# Walk through images
for root, _, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            print(f"Processing: {image_path}")

            try:
                # Load image and encode faces
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                for encoding in encodings:
                    face_db.append({
                        "path": os.path.relpath(image_path),  # relative path
                        "encoding": encoding
                    })

                # Generate thumbnail if not already present
                thumb_name = os.path.basename(image_path)
                thumb_path = os.path.join(THUMBNAIL_DIR, thumb_name)

                if not os.path.exists(thumb_path):
                    with Image.open(image_path) as img:
                        img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_WIDTH * 10000), Image.LANCZOS)
                        img.save(thumb_path)
                        print(f"Thumbnail created: {thumb_path}")
            except Exception as e:
                print(f"❌ Error with {image_path}: {e}")

# Save index
with open(OUTPUT_INDEX, "wb") as f:
    pickle.dump(face_db, f)

print(f"\n✅ Face index written to {OUTPUT_INDEX}")