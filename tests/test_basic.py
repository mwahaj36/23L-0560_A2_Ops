import io
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app, db
from app.models import InferenceLog

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"VisionGuard Deep Learning Detector" in response.data

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "healthy"
    assert json_data["database"] == "connected"

def test_invalid_file_extension(client):
    data = {"image": (io.BytesIO(b"dummy content"), "test.txt")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Invalid File Type" in response.data

def test_empty_upload(client):
    data = {"image": (io.BytesIO(b""), "")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
