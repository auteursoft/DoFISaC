import face_recognition
import pickle
import os

# Load the index
with open("face_index.pkl", "rb") as f:
    face_db = pickle.load(f)

# Load the query face
query_image = face_recognition.load_image_file("query.jpg")
query_encodings = face_recognition.face_encodings(query_image)

# Match results
results = []
for query_encoding in query_encodings:
    for entry in face_db:
        match = face_recognition.compare_faces([entry["encoding"]], query_encoding, tolerance=0.6)
        if match[0]:
            results.append(entry["path"])

# Generate HTML (referencing image files on disk)
html = ['<html><head><title>Face Match Results</title></head><body>']
html.append('<h1>Matched Images</h1>')

if results:
    for path in results:
        rel_path = os.path.relpath(path)  # ensures relative path from current dir
        html.append(f'<div><p>{rel_path}</p><img src="{rel_path}" style="max-width: 300px;"><br><br></div>')
else:
    html.append('<p>No matches found.</p>')

html.append('</body></html>')

# Write to HTML file
with open("matches.html", "w") as f:
    f.write('\n'.join(html))

print("HTML file with results saved to matches.html")