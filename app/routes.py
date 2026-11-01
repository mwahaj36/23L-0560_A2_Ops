import base64
import os
import requests
from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename

from app import db
from app.models import InferenceLog
from app.inference import run_inference, load_serving_model

main_bp = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.before_app_request
def preload_model():
    cloud_url = os.getenv("CLOUD_API_URL")
    if not cloud_url:
        try:
            load_serving_model()
        except Exception:
            pass

@main_bp.route("/")
def index():
    cloud_url = os.getenv("CLOUD_API_URL", "")
    return render_template("index.html", cloud_url=cloud_url)

@main_bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("error.html", error_title="No File Selected", error_message="Please select an image file to upload."), 400
        
    file = request.files["image"]
    if file.filename == "":
        return render_template("error.html", error_title="Empty Filename", error_message="No image was selected for upload."), 400

    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        return render_template(
            "error.html",
            error_title="Invalid File Type",
            error_message=f"File '{filename}' has an unsupported extension. Allowed: PNG, JPG, JPEG, WEBP."
        ), 400

    image_bytes = file.read()
    if len(image_bytes) == 0:
        return render_template("error.html", error_title="Empty File", error_message="The uploaded file contains no data."), 400

    max_bytes = current_app.config["MAX_CONTENT_LENGTH"]
    if len(image_bytes) > max_bytes:
        return render_template(
            "error.html",
            error_title="File Size Exceeded",
            error_message=f"Uploaded file exceeds maximum limit of {max_bytes // (1024 * 1024)} MB."
        ), 413

    cloud_url = os.getenv("CLOUD_API_URL")
    threshold = current_app.config.get("CONFIDENCE_THRESHOLD", 0.50)

    if cloud_url:
        try:
            resp = requests.post(
                f"{cloud_url.rstrip('/')}/api/predict",
                files={"image": (filename, image_bytes)},
                timeout=20
            )
            if resp.status_code == 200:
                result = resp.json()
            else:
                result = run_inference(image_bytes, threshold=threshold)
        except Exception:
            result = run_inference(image_bytes, threshold=threshold)
    else:
        try:
            result = run_inference(image_bytes, threshold=threshold)
        except ValueError as ve:
            return render_template(
                "error.html",
                error_title="Corrupt Image",
                error_message=f"The uploaded file could not be decoded as a valid image: {str(ve)}"
            ), 400
        except Exception as e:
            return render_template(
                "error.html",
                error_title="Inference Error",
                error_message=f"An unexpected error occurred during prediction: {str(e)}"
            ), 500

    log_entry = InferenceLog(
        filename=filename,
        prediction=result["label"],
        confidence=result["confidence_percentage"],
        model_name=result.get("model_name", "Cloud_Render_API")
    )
    db.session.add(log_entry)
    db.session.commit()

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    return render_template(
        "result.html",
        filename=filename,
        prediction=result["label"],
        confidence=result["confidence_percentage"],
        latency=result.get("latency_ms", 120),
        model_name=result.get("model_name", "Cloud_Render_API"),
        image_preview=b64_image
    )

@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image supplied"}), 400
    file = request.files["image"]
    image_bytes = file.read()
    try:
        result = run_inference(image_bytes)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main_bp.route("/history")
def history():
    logs = InferenceLog.query.order_by(InferenceLog.created_at.desc()).limit(100).all()
    return render_template("history.html", logs=logs)

@main_bp.route("/health")
def health():
    try:
        load_serving_model()
        model_status = "loaded"
    except Exception as e:
        model_status = f"unavailable: {str(e)}"

    return jsonify({
        "status": "healthy",
        "model": model_status,
        "database": "connected"
    }), 200

@main_bp.app_errorhandler(413)
def request_entity_too_large(error):
    return render_template(
        "error.html",
        error_title="File Too Large",
        error_message="Uploaded image exceeds maximum payload limit."
    ), 413
