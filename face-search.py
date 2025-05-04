import face_recognition
import pickle

# Load the index
with open("face_index.pkl", "rb") as f:
    face_db = pickle.load(f)

# Load the query face
query_image = face_recognition.load_image_file("query.jpg")
query_encodings = face_recognition.face_encodings(query_image)

results = []
for query_encoding in query_encodings:
    for entry in face_db:
        match = face_recognition.compare_faces([entry["encoding"]], query_encoding, tolerance=0.6)
        if match[0]:
            results.append(entry["path"])

print("Found matches:", results)