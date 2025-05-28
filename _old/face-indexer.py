import os
import sys
import face_recognition
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

def get_image_files(image_dir):
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
    return image_files

def process_file(filepath):
    try:
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        return [{"path": filepath, "encoding": enc} for enc in encodings]
    except Exception as e:
        return f"⚠️ Skipped {filepath}: {e}"

def main(image_dir, output_path="face_index.pkl", max_workers=None):
    image_files = get_image_files(image_dir)
    face_db = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file): file for file in image_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if isinstance(result, str):
                print(result)
            else:
                face_db.extend(result)

    with open(output_path, "wb") as f:
        pickle.dump(face_db, f)

    print(f"✅ Indexed {len(face_db)} faces from {len(image_files)} images.")
    print(f"📦 Saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_faces.py /path/to/images")
        sys.exit(1)

    image_dir = sys.argv[1]
    main(image_dir)