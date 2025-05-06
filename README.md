# Facial Search App Bundle

## Contents
- `flask-app/` - The full web UI with integrated feedback
- `retrain.py` - Cron-compatible retraining script
- `cronjob.txt` - Crontab entry to schedule daily retraining
- `logo.png` - Silly logo for branding

## Installation Steps
1. Unzip `flask-app` and install dependencies (see previous instructions)
2. Run the Flask app with `python app.py`
3. Use `face-indexer.py` to generate your vector indexes
4. Add the crontab entry from `cronjob.txt` to run `retrain.py` daily

Enjoy the selfies! 🪞
