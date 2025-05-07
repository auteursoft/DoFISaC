import os
import sys
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import hashlib
import multiprocessing
from tqdm import tqdm

THUMBNAIL_DIR = "static/thumbnails"
OUTPUT_PKL = "face_index.pkl"
ERROR_LOG = "index.err"

# Helper for generating consistent hash-based thumbnail names
def hash_filename(path):
    return hashlib.md5(path.encode()).hexdigest() + os.path.splitext(path)[1]

def process_image(filepath):
    try:
        from insightface.app import FaceAnalysis
        from transformers import CLIPProcessor, CLIPModel
        import torch

        face_model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        face_model.prepare(ctx_id=0)

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        img = Image.open(filepath).convert("RGB")
        img_np = np.array(img)

        faces = face_model.get(img_np)
        if not faces:
            raise ValueError("No face detected.")
        face_vec = faces[0].embedding

        inputs = clip_processor(images=img, return_tensors="pt", padding=True)
        bg_vec = clip_model.get_image_features(**inputs)[0].detach().numpy()

        thumb_name = hash_filename(filepath)
        thumb_path = os.path.join(THUMBNAIL_DIR, thumb_name)
        img.thumbnail((160, 160))
        img.save(thumb_path)

        return {
            "path": filepath,
            "thumb_name": thumb_name,
            "face_vec": face_vec,
            "bg_vec": bg_vec,
            "match": None,
            "distance": None,
            "feedback": None
        }

    except Exception as e:
        return {"error": f"{filepath} | {str(e)}"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python face-indexer.py <path_to_photos>")
        sys.exit(1)

    PHOTO_DIR = sys.argv[1]
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)

    image_paths = [str(p) for p in Path(PHOTO_DIR).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths), desc="Indexing photos"))

    face_db = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]

    if errors:
        with open(ERROR_LOG, "w") as ef:
            ef.write("\n".join(errors))

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(face_db, f)

    print(f"✅ Indexed {len(face_db)} images from {PHOTO_DIR}. Skipped {len(errors)}.")
