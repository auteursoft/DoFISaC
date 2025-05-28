import face_recognition
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def select_file(title="Select File", filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=filetypes)

def load_index(index_path):
    try:
        with open(index_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load face index: {e}")
        exit(1)

def search_faces(query_path, face_db, tolerance=0.6):
    try:
        query_image = face_recognition.load_image_file(query_path)
        encodings = face_recognition.face_encodings(query_image)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process query image: {e}")
        return []

    if not encodings:
        messagebox.showinfo("No Face Found", "No face found in query image.")
        return []

    results = []
    for encoding in encodings:
        for entry in face_db:
            if face_recognition.compare_faces([entry["encoding"]], encoding, tolerance=tolerance)[0]:
                results.append(entry["path"])

    return results

def main():
    print("📂 Select query image...")
    query_path = select_file("Select Query Image")
    if not query_path:
        print("❌ No image selected.")
        return

    print("📁 Select face index file...")
    index_path = select_file("Select face_index.pkl", filetypes=(("Pickle files", "*.pkl"),))
    if not index_path:
        print("❌ No index selected.")
        return

    print("🔍 Searching...")
    face_db = load_index(index_path)
    matches = search_faces(query_path, face_db)

    if matches:
        print("\n✅ Matches found:")
        for path in matches:
            print("→", path)
    else:
        print("\n❌ No matches found.")

if __name__ == "__main__":
    main()