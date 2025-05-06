# Facial Search App (Complete Bundle)

## Quick Start

1. Unzip:
   ```bash
   unzip facial_search_app_complete_ready.zip
   cd facial_search_app_complete_ready
   ```

2. Add `face_index.pkl` to the same folder.

3. Install dependencies:
   ```bash
   pip install flask insightface faiss-cpu transformers torch pillow tqdm
   ```

4. Run:
   ```bash
   python app.py
   ```

5. Visit: http://localhost:5000

## Retraining

- Feedback is saved per image in `static/feedback.json`
- Daily retraining is configured via `cronjob.txt`:
  ```bash
  crontab -e
  ```

Enjoy your high-performance, feedback-aware facial search engine!
