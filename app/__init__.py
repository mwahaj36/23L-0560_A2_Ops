import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "visionguard-secret-key-12345")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_FILE_SIZE_MB", 5)) * 1024 * 1024
    app.config["MODEL_PATH"] = os.getenv("MODEL_PATH", "models/champion_model.keras")
    app.config["CONFIDENCE_THRESHOLD"] = float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
    
    db.init_app(app)
    
    with app.app_context():
        from app.models import InferenceLog
        db.create_all()
        
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app

