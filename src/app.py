"""Local Flask UI and standalone cloud inference API."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, render_template, request
from sqlalchemy import create_engine, text
from werkzeug.utils import secure_filename

from inference import MAX_BYTES, decode_image, load_model, predict

WARNING = "This is a statistical estimate, not proof of an image's origin."


def create_app(config=None):
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.update(MAX_CONTENT_LENGTH=MAX_BYTES + 1024 * 1024,
                      API_URL=os.environ.get("API_URL", ""),
                      API_MODE=os.environ.get("API_MODE", "0") == "1",
                      DATABASE_URL=os.environ.get("DATABASE_URL", "sqlite:///history.db"))
    if config:
        app.config.update(config)
    engine = None
    if not app.config["API_MODE"]:
        engine = create_engine(app.config["DATABASE_URL"], pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("""CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, model_version TEXT NOT NULL,
                prediction TEXT NOT NULL, confidence REAL NOT NULL, filename TEXT NOT NULL,
                width INTEGER NOT NULL, height INTEGER NOT NULL)"""))
    elif not app.config.get("TESTING"):
        load_model()

    @app.get("/health")
    def health():
        return jsonify(status="healthy")

    @app.get("/")
    @app.get("/upload")
    def upload():
        return render_template("upload.html")

    @app.post("/predict")
    def prediction():
        uploaded = request.files.get("image")
        if uploaded is None:
            return _error("Please choose an image to upload.", app.config["API_MODE"])
        filename = secure_filename(uploaded.filename or "")
        try:
            data = uploaded.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                return _error("The image must be 5 MB or smaller.", app.config["API_MODE"], 413)
            image, (width, height) = decode_image(filename, data)
            if app.config["API_URL"] and not app.config["API_MODE"]:
                response = requests.post(app.config["API_URL"].rstrip("/") + "/predict",
                                         files={"image": (filename, data, uploaded.mimetype)}, timeout=60)
                response.raise_for_status()
                result = response.json()
            else:
                result = predict(image)
        except ValueError as error:
            return _error(str(error), app.config["API_MODE"])
        except (requests.RequestException, FileNotFoundError) as error:
            app.logger.error("Inference unavailable: %s", error)
            return _error("Inference is temporarily unavailable. Please try again later.", app.config["API_MODE"], 503)
        result["warning"] = WARNING
        if app.config["API_MODE"]:
            return jsonify(result)
        record = {"id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
                  "model_version": result["model_version"], "prediction": result["label"],
                  "confidence": result["confidence"], "filename": filename,
                  "width": width, "height": height}
        with engine.begin() as conn:
            conn.execute(text("""INSERT INTO predictions
                (id, timestamp, model_version, prediction, confidence, filename, width, height)
                VALUES (:id, :timestamp, :model_version, :prediction, :confidence, :filename, :width, :height)"""),
                         record)
        return render_template("result.html", result=result, filename=filename)

    @app.get("/history")
    def history():
        if app.config["API_MODE"]:
            return jsonify(error="No history in API mode"), 404
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100")).mappings().all()
        return render_template("history.html", rows=rows)

    @app.errorhandler(413)
    def too_large(_error_info):
        return _error("The image must be 5 MB or smaller.", app.config["API_MODE"], 413)

    return app


def _error(message, api_mode, status=400):
    if api_mode:
        return jsonify(error=message), status
    return render_template("error.html", message=message), status


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
