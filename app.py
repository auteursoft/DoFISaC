from flask import Flask, render_template
from pathlib import Path

app = Flask(__name__)

@app.route("/")
def index():
    thumb_dir = Path("static/thumbnails")
    images = sorted([f.name for f in thumb_dir.glob("*") if f.is_file()])
    return render_template("test_thumbnails.html", images=images)

if __name__ == "__main__":
    app.run(debug=True)
