import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from insightface.app import FaceAnalysis
from transformers import CLIPProcessor, CLIPModel

PHOTO_DIR = "photos"
OUTPUT_PKL = "face_index.pkl"
THUMBNAIL_DIR = "static/thumbnails"

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

face_db = []
for root, _, files in os.walk(PHOTO_DIR):
    for file in tqdm(files):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        filepath = os.path.join(root, file)
        try:
            img = Image.open(filepath).convert("RGB")
            img_np = np.array(img)
            face_vec = extract_face_embedding(img_np)
            bg_vec = extract_clip_embedding(img)
            thumb_name = os.path.basename(filepath)
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
            print(f"Skipped {file}: {e}")

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(face_db, f)
print(f"✅ Indexed {len(face_db)} images.")
