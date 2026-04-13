from __future__ import annotations

import argparse
import base64
import binascii
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "ai-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pose_service import pose_service


def _decode_image(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img
    try:
        text = data.decode("utf-8").strip()
    except Exception:
        return None
    if not text:
        return None
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1].strip()
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        try:
            raw = base64.b64decode(text + "=" * ((4 - len(text) % 4) % 4))
        except Exception:
            return None
    arr2 = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr2, cv2.IMREAD_UNCHANGED)


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {getattr(image, 'shape', None)}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test local pose inference on one image")
    parser.add_argument("image_path", help="Path to image file")
    args = parser.parse_args()

    p = Path(args.image_path)
    data = p.read_bytes()
    img = _decode_image(data)
    if img is None:
        print("decode_failed")
        return 2
    rgb = _normalize_rgb(img)

    ok = pose_service.init()
    print(f"pose_initialized={ok}")
    detector = pose_service.get_detector(request_id="cli-test")
    result = detector.process(rgb)
    landmarks = result.pose_landmarks.landmark if (result and result.pose_landmarks) else []
    print(f"landmarks_count={len(landmarks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
