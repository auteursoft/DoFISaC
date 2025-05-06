# Facial-Search

This version avoids thumbnail collision and handles special characters. 

<iframe width="560" height="315" src="https://www.youtube.com/watch?v=raRGnueg8Lo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

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

