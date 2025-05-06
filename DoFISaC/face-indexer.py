import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import hashlib
from insightface.app import FaceAnalysis
from transformers import CLIPProcessor, CLIPModel

if len(sys.argv) < 2:
    print("❌ Usage: python face-indexer.py <path_to_photos>")
    sys.exit(1)

PHOTO_DIR = sys.argv[1]
OUTPUT_PKL = "face_index.pkl"
THUMBNAIL_DIR = "static/thumbnails"
ERROR_LOG = "index.err"

os.makedirs(THUMBNAIL_DIR, exist_ok=True)

face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_clip_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    return clip_model.get_image_features(**inputs)[0].detach().numpy()

def extract_face_embedding(image_np):
    faces = face_model.get(image_np)
    return faces[0].embedding if faces else None

def hash_filename(path):
    return hashlib.md5(path.encode()).hexdigest() + os.path.splitext(path)[1]

face_db = []
errors = []

image_paths = [str(p) for p in Path(PHOTO_DIR).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

for filepath in tqdm(image_paths, desc="Indexing photos"):
    try:
        img = Image.open(filepath).convert("RGB")
        img_np = np.array(img)

        face_vec = extract_face_embedding(img_np)
        if face_vec is None:
            raise ValueError("No face detected.")

        bg_vec = extract_clip_embedding(img)
        if bg_vec is None:
            raise ValueError("Failed to compute background vector.")

        thumb_name = hash_filename(filepath)
        thumb_path = os.path.join(THUMBNAIL_DIR, thumb_name)
        img.thumbnail((160, 160))
        img.save(thumb_path)

        face_db.append({
            "path": filepath,
            "thumb_name": thumb_name,
            "face_vec": face_vec,
            "bg_vec": bg_vec
        })

    except Exception as e:
        errors.append(f"{filepath} | {str(e)}")

# Save error log
if errors:
    with open(ERROR_LOG, "w") as ef:
        ef.write("\n".join(errors))

# Save index
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(face_db, f)

print(f"✅ Indexed {len(face_db)} images from {PHOTO_DIR}. Skipped {len(errors)}.")
