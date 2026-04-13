"""Last ned MediaPipe Tasks-modeller (.task) til models/ ved behov."""
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
OPTIONAL_MODELS = {
    # OpenCV Zoo SFace (ArcFace-like face embedding model).
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def ensure_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        path = MODELS_DIR / name
        if path.exists():
            continue
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise RuntimeError(f"Kunne ikke laste ned {name} fra {url}: {e}") from e


def get_face_model_path() -> str:
    ensure_models()
    return str(MODELS_DIR / "face_landmarker.task")


def get_pose_model_path() -> str:
    ensure_models()
    return str(MODELS_DIR / "pose_landmarker.task")


def _ensure_optional_model(name: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = OPTIONAL_MODELS.get(name)
    if not url:
        raise RuntimeError(f"Ukjent optional modell: {name}")
    path = MODELS_DIR / name
    if not path.exists():
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise RuntimeError(f"Kunne ikke laste ned {name} fra {url}: {e}") from e
    return str(path)


def get_face_embedding_model_path() -> str:
    return _ensure_optional_model("face_recognition_sface_2021dec.onnx")
