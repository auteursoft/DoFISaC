# facial-search
Utility for searching by face on OSX (And possibly other OS's)

**Works with python3.11 on OSX. As of this commit, did not work with higher versions**. 

## Installation
1. Configure python 3 virtual environment: `python3.11 -m venv .venv`
2. Source the virtual env: `source .venv/bin/activate`
3. Install required libraries: 
    - Precise `pip install -r requirements.txt`
    - If you add libraries in a fork, you can recreate the `requirements.txt` file with: 

    ```bash
    pip freeze | cut -d '=' -f 1 > requirements.txt
    ```

## Indexing: 
1. `python index-and-cluster.py "/Volumes/super_54/google/sean.goggins/Google Photos"` for example. 


## Docker Instructions: 
1. Create a file with a list of directories to scan, like this: 
```bash
/data/photos_2022
/data/photos_2023
```
2. `docker build -t face-indexer-multi .` 
3. Run The Container: 
```bash
docker run --rm \
  -v "$(pwd)/config:/config" \
  -v "$(pwd)/output:/output" \
  -v "$(pwd)/data:/data" \
  face-indexer-multi
```

## Desktop App (Still working on it)
1. Ensure you have Python 3.11 installed.
2. From Terminal, run:
```bash
   chmod +x run_search.command
   ./run_search.command
```

This will open a file dialog to choose your query image and face index (face_index.pkl).

To turn this into a native app:
- Use Automator on macOS: create an "Application" that runs `run_search.command`.

## Web App (Alpha)
1. 🚨 This is now a native web app that usees Flask
2. ... 

🚨 There are different `pip install` requirements for the web app, FYI. 

© Sean P. Goggins, all rights reserved, 2025
