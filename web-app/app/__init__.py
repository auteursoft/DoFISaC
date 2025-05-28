from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    app.config['INDEX_FOLDER'] = 'app/static/indexed_faces'

    from .routes import main
    app.register_blueprint(main)

    return app
