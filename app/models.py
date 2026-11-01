from datetime import datetime
from app import db

class InferenceLog(db.Model):
    __tablename__ = "inference_logs"
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

