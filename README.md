# facial-search
Utility for searching by face on OSX (And possibly other OS's)

**Works with python3.11 on OSX. As of this commit, did not work with higher versions**. 

## Installation
1. Configure python 3 virtual environment: `python3.11 -m venv .venv`
2. Source the virtual env: `source .venv/bin/activate`
3. Install required libraries: 
    - Safe: `pip install opencv-python face_recognition tqdm` 
    - Precise `pip install -r requirements.txt`

## Indexing: 
1. `python face-indexer.py "/Volumes/super_54/google/sean.goggins/Google Photos"` for example. 


## Docker Instructions: 
1. `docker build -t facial-search .` 
2. Run The Container: 
```docker 
    docker run --rm \
    -v /Users/sean/Pictures:/data \
    -v $(pwd):/app \
    facial-search
```