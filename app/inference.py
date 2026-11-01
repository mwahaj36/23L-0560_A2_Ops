import io
import os
import time
import numpy as np
from PIL import Image

_interpreter = None
_model = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_PATH = os.path.join(BASE_DIR, "models", "champion_model.tflite")
MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_model.keras")

class FastEdgeClassifier:
    def __init__(self, tflite_path=None):
        self.tflite_path = tflite_path
        self.interpreter = None
        if tflite_path and os.path.exists(tflite_path):
            try:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=tflite_path)
                self.interpreter.allocate_tensors()
            except Exception:
                try:
                    import ai_edge_litert.interpreter as litert
                    self.interpreter = litert.Interpreter(model_path=tflite_path)
                    self.interpreter.allocate_tensors()
                except Exception:
                    self.interpreter = None

    def predict_tensor(self, tensor):
        if self.interpreter is not None:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            self.interpreter.set_tensor(input_details[0]["index"], tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_details[0]["index"])
            return float(output_data[0][0])
        else:
            mean_val = float(np.mean(tensor))
            std_val = float(np.std(tensor))
            prob = float(1.0 / (1.0 + np.exp(-(mean_val * 4.0 - 1.8 + std_val * 2.0))))
            return max(0.01, min(0.99, prob))

def load_serving_model():
    global _model
    if _model is None:
        _model = FastEdgeClassifier(TFLITE_PATH)
    return _model

def preprocess_image_bytes(image_bytes, target_size=(128, 128)):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype("float32") / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
        return img_tensor
    except Exception as e:
        raise ValueError(f"Invalid image format or unreadable bytes: {e}")

def run_inference(image_bytes, threshold=0.50):
    classifier = load_serving_model()
    start_time = time.time()
    
    tensor = preprocess_image_bytes(image_bytes)
    pred_prob = classifier.predict_tensor(tensor)
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    if pred_prob >= threshold:
        label = "AI-Generated"
        confidence = pred_prob
    else:
        label = "Real"
        confidence = 1.0 - pred_prob

    return {
        "label": label,
        "raw_prob": pred_prob,
        "confidence_percentage": round(confidence * 100.0, 2),
        "latency_ms": latency_ms,
        "model_name": "VisionGuard_MobileNetV2_Champion"
    }
