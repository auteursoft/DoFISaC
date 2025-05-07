# DoFISaC: Division of Facial Image Searching and Clustering

This version avoids thumbnail collision and handles special characters. 

I would not rick roll you with an image of a rick roll. Ergo, this is called a reverse rick roll, or in some quarters a double secret rick roll. 

[![Watch the video](https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg)](https://www.youtube.com/watch?v=raRGnueg8Lo)

## Features
1. ✅ InsightFace + CLIP
2. ✅ FAISS search
3. ✅ Binary voting for face matches
4. ✅ 1–6 scale voting for cluster quality

### Feature Descriptions
1.	clustering.py
 - Uses only thumbnail references, no full-size image copying.
 - Outputs proper phash_clusters.json and bg_clusters.json with thumbnail names and original paths.
 - Fully multiprocessing-safe.
2.	Templates (index.html, search.html, clusters_phash.html, clusters_bg.html):
 - Proper logo references, styles, and structure.
 - Pagination support and working lazy loading.
 - Face search results display:
 - Correct Match and Distance values.
 - Feedback controls.
 - Clickable full-size photo links (using href).
3.	app.py:
 - Matches all template expectations (e.g., page_count defined).
 - All endpoints return expected data for templates.
 - Supports search pagination and feedback routing.
4.	face-indexer.py:
 - Proper Match and Distance calculation and formatting.
 - Logs errors cleanly.
 - Generates thumbnails and stores only references.
5.	Search result logic:
 - Feedback UI fully embedded.
 - Links function properly to full-size files.

## Specific Scripts and Directories (Under the DoFISaC/ Directory): 
6. face-indexer.py – Embeds faces and scenes, generates thumbnails, builds index `python face-indexer.py "/path/to/my/photo/folder"`
7. clustering.py – Combines vectors, clusters using DBSCAN based on data scale
8. retrain.py – Placeholder for retraining logic
9. cronjob.txt – Example for scheduled retraining
10. templates/ – Basic HTML for search, results, cluster feedback
11. static/ – Logo and stylesheet
12. README.md – Basic setup instructions
13. `retry-errors.py` will scan the error log and add previously errored files to an existing face-search pkl file. Clustering will need to be redone entirely if the clusters have already been generated. 


## Installation:
1. Create a python 3 virutal environment using python 3.11. Versions of python above 3.11 do not support some of the libraries used here at the time of creation. The `requirements.txt` file is library version agnostic in order to better support future updates without repository modification. For research and development always be sure to save the specific versions used for future reference by running something like `pip freeze > 20250506-execution.txt`, for example. 

```bash
python 3.11 -m venv .venv
```
2. Install libraries
```bash
pip install -r requirements.txt
```
3. Activate the venv: 
```bash
source .venv/bin/activate
```


## Saving updated requirements without versioning (i.e., if you have an updated PR to submit or something similar):
1. Unix/OSX: `pip freeze | cut -d '=' -f 1 > requirements.txt`
2. Windows Powershell `pip freeze | ForEach-Object { ($_ -split '==')[0] } > requirements.txt`
3. Full toxic pythonity: `python -m pip list --format=freeze | cut -d '=' -f 1 > requirements.txt`

