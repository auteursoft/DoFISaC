import os
import pickle
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import hashlib

ERROR_LOG = "index.err"
INDEX_FILE = "face_index.pkl"
THUMBNAIL_DIR = "static/thumbnails"

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

# Load existing index
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = []

# Read error log
if not os.path.exists(ERROR_LOG):
    print("No error log found.")
    exit(1)

with open(ERROR_LOG, "r") as ef:
    lines = [line.strip() for line in ef.readlines() if line.strip()]

retry_paths = [line.split(" | ")[0] for line in lines if os.path.exists(line.split(" | ")[0])]

successful = 0
remaining_errors = []

for filepath in tqdm(retry_paths):
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
        successful += 1

    except Exception as e:
        remaining_errors.append(f"{filepath} | {str(e)}")

with open(INDEX_FILE, "wb") as f:
    pickle.dump(face_db, f)

with open(ERROR_LOG, "w") as ef:
    ef.write("\n".join(remaining_errors))

print(f"✅ Recovered {successful} files. Remaining errors: {len(remaining_errors)}")
