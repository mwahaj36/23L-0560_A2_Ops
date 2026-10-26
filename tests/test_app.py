import io

from PIL import Image

import app


def make_image():
    stream = io.BytesIO()
    Image.new("RGB", (20, 10), "#3050a0").save(stream, "PNG")
    stream.seek(0)
    return stream


def test_health_and_validation(tmp_path):
    client = app.create_app({"TESTING": True, "DATABASE_URL": "sqlite:///" + str(tmp_path / "history.db")}).test_client()
    assert client.get("/health").json == {"status": "healthy"}
    assert client.post("/predict", data={"image": (io.BytesIO(b"bad"), "bad.png")}).status_code == 400
    assert client.post("/predict", data={"image": (make_image(), "bad.txt")}).status_code == 400
    assert client.post("/predict", data={"image": (io.BytesIO(b"a" * (5 * 1024 * 1024 + 1)), "big.png")}).status_code == 413


def test_prediction_persists_without_upload_file(tmp_path, monkeypatch):
    result = {"label": "Real", "confidence": 91.0, "probability_ai": 0.09,
              "threshold": 0.5, "model_version": "test-v1"}
    monkeypatch.setattr(app, "predict", lambda image: result)
    client = app.create_app({"TESTING": True, "DATABASE_URL": "sqlite:///" + str(tmp_path / "history.db")}).test_client()
    response = client.post("/predict", data={"image": (make_image(), "sample.png")})
    assert response.status_code == 200
    assert b"91.0%" in response.data
    assert b"sample.png" in client.get("/history").data
    assert sorted(path.name for path in tmp_path.iterdir()) == ["history.db"]
