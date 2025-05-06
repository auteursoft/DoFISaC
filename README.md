# DoFISaC: Division of Facial Image Searching and Clustering

This version avoids thumbnail collision and handles special characters. 

I would not rick roll you with an image of a rick roll. Ergo, this is called a reverse rick roll, or in some quarters a double secret rick roll. 

[![Watch the video](https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg)](https://www.youtube.com/watch?v=raRGnueg8Lo)

1. app.py – Flask app with:
2. ✅ InsightFace + CLIP
3. ✅ FAISS search
4. ✅ Binary voting for face matches
5. ✅ 1–6 scale voting for cluster quality
6. face-indexer.py – Embeds faces and scenes, generates thumbnails, builds index
7. clustering.py – Combines vectors, clusters using DBSCAN based on data scale
8. retrain.py – Placeholder for retraining logic
9. cronjob.txt – Example for scheduled retraining
10. templates/ – Basic HTML for search, results, cluster feedback
11. static/ – Logo and stylesheet
12. README.md – Basic setup instructions
13. `retry-errors.py` will scan the error log and add previously errored files to an existing face-search pkl file. Clustering will need to be redone entirely if the clusters have already been generated. 
14. Love

## Installation:



## Saving updated requirements:
1. Unix/OSX: `pip freeze | cut -d '=' -f 1 > requirements.txt`
2. Windows Powershell `pip freeze | ForEach-Object { ($_ -split '==')[0] } > requirements.txt`
3. Full toxic pythonity: `python -m pip list --format=freeze | cut -d '=' -f 1 > requirements.txt`

