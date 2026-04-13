from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh
from PIL import Image

logger = logging.getLogger(__name__)
HEAD_KWS = ("head", "face", "neck")
HAIR_KWS = ("hair", "scalp", "brow", "eyebrow", "beard")
EYE_KWS = ("eye", "iris", "cornea", "sclera", "pupil")
DEBUG_FORCE_TEXTURE = os.getenv("DEBUG_FORCE_TEXTURE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _weight_smooth(raw: float) -> float:
    # Smooth morph weights to avoid extreme distortion.
    return _clamp(0.5 + float(raw) * 0.30, 0.2, 0.8)


def _pt(points: np.ndarray, idx: int) -> Optional[np.ndarray]:
    if points is None or len(points) <= idx:
        return None
    p = points[idx]
    if p is None or len(p) < 2:
        return None
    return np.asarray([float(p[0]), float(p[1])], dtype=np.float32)


def _dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _safe_ratio(a: Optional[float], b: Optional[float], default: float) -> float:
    if a is None or b is None:
        return float(default)
    if abs(float(b)) < 1e-6:
        return float(default)
    return float(a) / float(b)


def _median_rgb(patch_rgb: np.ndarray) -> Optional[tuple[int, int, int]]:
    if patch_rgb is None or patch_rgb.size == 0:
        return None
    vals = np.median(patch_rgb.reshape(-1, 3), axis=0)
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def _crop(rgb: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
    if rgb is None or rgb.size == 0:
        return None
    h, w = rgb.shape[:2]
    xi1 = int(max(0, min(w - 1, round(float(x1)))))
    yi1 = int(max(0, min(h - 1, round(float(y1)))))
    xi2 = int(max(0, min(w, round(float(x2)))))
    yi2 = int(max(0, min(h, round(float(y2)))))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    out = rgb[yi1:yi2, xi1:xi2]
    if out is None or out.size == 0:
        return None
    return out


def default_face_params() -> dict:
    return {
        "faceWidthHeight": 0.78,
        "jawWidthRatio": 0.78,
        "cheekboneWidthRatio": 0.86,
        "foreheadHeightRatio": 0.30,
        "interocularRatio": 0.39,
        "eyeSizeRatio": 0.16,
        "noseWidthRatio": 0.24,
        "noseLengthRatio": 0.27,
        "mouthWidthRatio": 0.36,
    }


def default_face_metrics() -> dict:
    return {
        "faceWidth": 0.78,
        "jawWidth": 0.60,
        "cheekboneWidth": 0.66,
        "eyeDistance": 0.30,
        "eyeSize": 0.11,
        "eyebrowHeight": 0.07,
        "eyebrowAngle": 0.00,
        "noseWidth": 0.19,
        "noseLength": 0.27,
        "mouthWidth": 0.28,
        "lipThickness": 0.05,
        "chinLength": 0.20,
        "faceHeightRatio": 1.30,
        "skullWidth": 0.70,
        "jawTaper": 0.90,
        "cheekboneProminence": 1.08,
        "foreheadHeight": 0.30,
        "eyelidOpenness": 0.10,
    }


def apply_face_morphs(landmark_metrics: dict, avatar_style: Optional[str] = None) -> dict:
    """
    Convert normalized face metrics -> semantic avatar morph controls in [-1, 1].
    """
    m = dict(default_face_metrics())
    if isinstance(landmark_metrics, dict):
        for k, v in landmark_metrics.items():
            try:
                m[k] = float(v)
            except Exception:
                continue

    ref = default_face_metrics()

    def rel(metric_name: str, strength: float = 1.0, invert: bool = False) -> float:
        base = float(ref.get(metric_name, 1.0))
        cur = float(m.get(metric_name, base))
        raw = (cur - base) / max(abs(base), 1e-6)
        if invert:
            raw *= -1.0
        return _clamp(raw * float(strength), -1.0, 1.0)

    style = str(avatar_style or "").strip().lower()
    female_boost = 1.22 if style == "female" else 1.0

    morphs = {
        "eyeSpacing": rel("eyeDistance", strength=1.25 * female_boost),
        "eyePosition": rel("eyebrowHeight", strength=0.70 * female_boost),
        "eyeSize": rel("eyeSize", strength=1.18 * female_boost),
        "noseWidth": rel("noseWidth", strength=1.35 * female_boost),
        "noseLength": rel("noseLength", strength=1.25 * female_boost),
        "jawWidth": rel("jawWidth", strength=1.10 * female_boost),
        "cheekboneWidth": rel("cheekboneWidth", strength=1.10 * female_boost),
        "chinHeight": rel("chinLength", strength=1.20 * female_boost),
        "faceHeight": rel("faceHeightRatio", strength=1.00 * female_boost),
        "mouthWidth": rel("mouthWidth", strength=1.30 * female_boost),
        "lipsFullness": rel("lipThickness", strength=1.55 * female_boost),
        "skullWidth": rel("skullWidth", strength=1.05 * female_boost),
        "jawTaper": rel("jawTaper", strength=1.05 * female_boost),
        "cheekboneProminence": rel("cheekboneProminence", strength=1.10 * female_boost),
        "foreheadHeight": rel("foreheadHeight", strength=1.00 * female_boost),
        "eyelidOpenness": rel("eyelidOpenness", strength=1.25 * female_boost),
        "eyebrowHeight": rel("eyebrowHeight", strength=1.25 * female_boost),
        "eyebrowAngle": _clamp(float(m.get("eyebrowAngle", 0.0)) * 3.8, -1.0, 1.0),
    }
    return {k: _clamp(float(v), -1.0, 1.0) for k, v in morphs.items()}


_face_embedder = "unset"


def _get_face_embedder():
    global _face_embedder
    if _face_embedder != "unset":
        return _face_embedder
    try:
        from models_download import get_face_embedding_model_path

        model_path = get_face_embedding_model_path()
        if hasattr(cv2, "FaceRecognizerSF_create"):
            _face_embedder = cv2.FaceRecognizerSF_create(str(model_path), "")
            return _face_embedder
    except Exception as e:
        logger.info("face embedding model unavailable: %s", e)
    _face_embedder = None
    return None


def _align_face_for_embedding(face_rgb: np.ndarray, landmarks_xy_px: Optional[np.ndarray], size: int = 112) -> Optional[np.ndarray]:
    if face_rgb is None or face_rgb.size == 0:
        return None
    if landmarks_xy_px is None or len(landmarks_xy_px) < 120:
        try:
            return cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        except Exception:
            return None
    pts = np.asarray(landmarks_xy_px, dtype=np.float32)
    l_eye = _pt(pts, 33)
    r_eye = _pt(pts, 263)
    nose = _pt(pts, 1)
    if l_eye is None or r_eye is None or nose is None:
        try:
            return cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        except Exception:
            return None
    src_tri = np.float32([l_eye, r_eye, nose])
    dst_tri = np.float32(
        [
            [size * 0.34, size * 0.38],
            [size * 0.66, size * 0.38],
            [size * 0.50, size * 0.58],
        ]
    )
    try:
        m = cv2.getAffineTransform(src_tri, dst_tri)
        return cv2.warpAffine(face_rgb, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    except Exception:
        try:
            return cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        except Exception:
            return None


def _simulate_avatar_front_face(aligned_face_rgb: np.ndarray, morph_params: dict) -> np.ndarray:
    img = aligned_face_rgb.astype(np.uint8)
    h, w = img.shape[:2]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    mp = dict(morph_params or {})
    sx = 1.0 + 0.05 * float(mp.get("jawWidth", 0.0)) + 0.04 * float(mp.get("skullWidth", 0.0)) + 0.03 * float(mp.get("eyeSpacing", 0.0))
    sy = 1.0 + 0.04 * float(mp.get("faceHeight", 0.0)) + 0.04 * float(mp.get("chinHeight", 0.0)) + 0.03 * float(mp.get("noseLength", 0.0))
    sx = _clamp(sx, 0.88, 1.14)
    sy = _clamp(sy, 0.88, 1.14)
    m = np.array([[sx, 0.0, (1.0 - sx) * cx], [0.0, sy, (1.0 - sy) * cy]], dtype=np.float32)
    warped = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    eye_open = _clamp(float(mp.get("eyelidOpenness", 0.0)), -1.0, 1.0)
    if abs(eye_open) > 1e-3:
        k = 0.7 + 0.4 * max(0.0, eye_open)
        y1, y2 = int(h * 0.26), int(h * 0.52)
        roi = warped[y1:y2, :, :]
        if roi.size > 0:
            roi2 = cv2.resize(roi, (roi.shape[1], max(2, int(round(roi.shape[0] * k)))), interpolation=cv2.INTER_CUBIC)
            if roi2.shape[0] < roi.shape[0]:
                pad = roi.shape[0] - roi2.shape[0]
                top = pad // 2
                roi2 = cv2.copyMakeBorder(roi2, top, pad - top, 0, 0, borderType=cv2.BORDER_REFLECT_101)
            warped[y1:y2, :, :] = roi2[: roi.shape[0], : roi.shape[1], :]
    return warped


def _embedding_vector(face_rgb_112: np.ndarray) -> Optional[np.ndarray]:
    embedder = _get_face_embedder()
    if embedder is None or face_rgb_112 is None or face_rgb_112.size == 0:
        return None
    try:
        bgr = cv2.cvtColor(face_rgb_112.astype(np.uint8), cv2.COLOR_RGB2BGR)
        vec = embedder.feature(bgr)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(arr))
        if n < 1e-8:
            return None
        return arr / n
    except Exception:
        return None


def validate_likeness_and_refine(
    face_metrics: dict,
    initial_morph_params: dict,
    face_rgb: Optional[np.ndarray] = None,
    landmarks_xy_px: Optional[np.ndarray] = None,
    avatar_style: Optional[str] = None,
    threshold: float = 0.86,
    max_iters: int = 4,
) -> dict:
    """
    Likeness validation with image embedding when available.
    Falls back to metric-space proxy if embedding model is unavailable.
    """
    if not isinstance(face_metrics, dict):
        return {
            "likenessScore": 0.0,
            "threshold": float(threshold),
            "iterations": 0,
            "improved": False,
            "bestMorphParameters": dict(initial_morph_params or {}),
            "validationMethod": "proxy_embedding_failed",
            "embeddingEnabled": False,
        }

    ref = default_face_metrics()
    metrics_order = [
        "faceWidth",
        "jawWidth",
        "cheekboneWidth",
        "eyeDistance",
        "eyeSize",
        "eyebrowHeight",
        "eyebrowAngle",
        "noseWidth",
        "noseLength",
        "mouthWidth",
        "lipThickness",
        "chinLength",
        "faceHeightRatio",
        "skullWidth",
        "jawTaper",
        "cheekboneProminence",
        "foreheadHeight",
        "eyelidOpenness",
    ]

    style = str(avatar_style or "").strip().lower()
    female_boost = 1.18 if style == "female" else 1.0

    morph_to_metric = {
        "eyeSpacing": ("eyeDistance", 0.35 * female_boost),
        "eyePosition": ("eyebrowHeight", 0.20),
        "eyeSize": ("eyeSize", 0.34 * female_boost),
        "noseWidth": ("noseWidth", 0.40 * female_boost),
        "noseLength": ("noseLength", 0.38 * female_boost),
        "jawWidth": ("jawWidth", 0.32 * female_boost),
        "cheekboneWidth": ("cheekboneWidth", 0.33 * female_boost),
        "chinHeight": ("chinLength", 0.34 * female_boost),
        "faceHeight": ("faceHeightRatio", 0.22),
        "mouthWidth": ("mouthWidth", 0.34 * female_boost),
        "lipsFullness": ("lipThickness", 0.42 * female_boost),
        "skullWidth": ("skullWidth", 0.26),
        "jawTaper": ("jawTaper", 0.28),
        "cheekboneProminence": ("cheekboneProminence", 0.28),
        "foreheadHeight": ("foreheadHeight", 0.24),
        "eyelidOpenness": ("eyelidOpenness", 0.36),
        "eyebrowHeight": ("eyebrowHeight", 0.30),
        "eyebrowAngle": ("eyebrowAngle", 0.22),
    }

    def embed_metrics(v: dict) -> np.ndarray:
        arr = np.asarray([float(v.get(k, ref.get(k, 0.0))) for k in metrics_order], dtype=np.float32)
        arr -= np.mean(arr)
        n = float(np.linalg.norm(arr))
        if n < 1e-8:
            return np.zeros_like(arr)
        return arr / n

    def morph_to_projected_metrics(morphs: dict) -> dict:
        projected = dict(ref)
        for mk, (fk, gain) in morph_to_metric.items():
            mv = _clamp(float((morphs or {}).get(mk, 0.0)), -1.0, 1.0)
            projected[fk] = float(ref.get(fk, 0.0)) * (1.0 + float(gain) * mv)
        return projected

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    source_aligned = _align_face_for_embedding(face_rgb, landmarks_xy_px, size=112) if face_rgb is not None else None
    source_vec = _embedding_vector(source_aligned) if source_aligned is not None else None
    embedding_enabled = bool(source_vec is not None)

    target_metrics = dict(ref)
    target_metrics.update({k: float(v) for k, v in face_metrics.items() if k in ref})
    target_emb = embed_metrics(target_metrics)
    current = {k: _clamp(float(v), -1.0, 1.0) for k, v in dict(initial_morph_params or {}).items()}
    for mk in morph_to_metric.keys():
        current.setdefault(mk, 0.0)

    def combined_score(morphs: dict) -> tuple[float, float, float]:
        metric_score = cosine(target_emb, embed_metrics(morph_to_projected_metrics(morphs)))
        image_score = 0.0
        if embedding_enabled and source_aligned is not None and source_vec is not None:
            avatar_proxy = _simulate_avatar_front_face(source_aligned, morphs)
            avatar_vec = _embedding_vector(avatar_proxy)
            if avatar_vec is not None:
                image_score = cosine(source_vec, avatar_vec)
        if embedding_enabled:
            total = 0.45 * metric_score + 0.55 * image_score
        else:
            total = metric_score
        return float(total), float(metric_score), float(image_score)

    best = dict(current)
    best_total, best_metric, best_image = combined_score(best)
    iters_used = 0
    if best_total >= float(threshold):
        return {
            "likenessScore": round(float(best_total), 4),
            "likenessScoreMetric": round(float(best_metric), 4),
            "likenessScoreEmbedding": round(float(best_image), 4),
            "threshold": float(threshold),
            "iterations": int(iters_used),
            "improved": False,
            "bestMorphParameters": best,
            "validationMethod": "sface_embedding+metric" if embedding_enabled else "proxy_front_embedding",
            "embeddingEnabled": embedding_enabled,
        }

    step_schedule = (0.24, 0.16, 0.10, 0.07, 0.05) if style == "female" else (0.18, 0.12, 0.08, 0.06)
    for step in step_schedule:
        if iters_used >= int(max_iters):
            break
        candidate = dict(best)
        for mk, (fk, gain) in morph_to_metric.items():
            base = float(ref.get(fk, 1.0))
            tgt = float(target_metrics.get(fk, base))
            desired = (tgt / max(1e-6, base) - 1.0) / max(1e-6, float(gain))
            desired = _clamp(desired, -1.0, 1.0)
            candidate[mk] = _clamp(float(candidate.get(mk, 0.0)) + (desired - float(candidate.get(mk, 0.0))) * step, -1.0, 1.0)
        score_total, score_metric, score_image = combined_score(candidate)
        iters_used += 1
        if score_total > best_total:
            best = candidate
            best_total = score_total
            best_metric = score_metric
            best_image = score_image
        if best_total >= float(threshold):
            break

    baseline_total, _, _ = combined_score(current)
    return {
        "likenessScore": round(float(best_total), 4),
        "likenessScoreMetric": round(float(best_metric), 4),
        "likenessScoreEmbedding": round(float(best_image), 4),
        "threshold": float(threshold),
        "iterations": int(iters_used),
        "improved": bool(best_total > baseline_total),
        "bestMorphParameters": best,
        "validationMethod": "sface_embedding+metric" if embedding_enabled else "proxy_front_embedding",
        "embeddingEnabled": embedding_enabled,
    }


def analyze_face_landmarks(face_rgb: np.ndarray, landmarks_xy_px: np.ndarray) -> dict:
    """
    Deterministic face ratios from MediaPipe FaceMesh landmarks.
    Landmarks are expected in pixel coordinates [N,2].
    """
    out = {
        "faceParams": default_face_params(),
        "faceMetrics": default_face_metrics(),
        "avatarMorphParameters": apply_face_morphs(default_face_metrics()),
        "faceColorRgb8": None,
        "faceLandmarksCount": int(len(landmarks_xy_px)) if landmarks_xy_px is not None else 0,
        "faceAnalysisSource": "failed",
    }
    if face_rgb is None or face_rgb.size == 0 or landmarks_xy_px is None or len(landmarks_xy_px) < 100:
        return out
    pts = np.asarray(landmarks_xy_px, dtype=np.float32)
    h, w = face_rgb.shape[:2]
    try:
        # Core points (MediaPipe FaceMesh indices).
        chin = _pt(pts, 152)
        forehead = _pt(pts, 10)
        left_face = _pt(pts, 234)
        right_face = _pt(pts, 454)
        left_jaw = _pt(pts, 172)
        right_jaw = _pt(pts, 397)
        left_cheek = _pt(pts, 93)
        right_cheek = _pt(pts, 323)
        left_eye_outer = _pt(pts, 33)
        right_eye_outer = _pt(pts, 263)
        left_eye_inner = _pt(pts, 133)
        right_eye_inner = _pt(pts, 362)
        left_eye_top = _pt(pts, 159)
        left_eye_bottom = _pt(pts, 145)
        right_eye_top = _pt(pts, 386)
        right_eye_bottom = _pt(pts, 374)
        nose_left = _pt(pts, 129)
        nose_right = _pt(pts, 358)
        nose_bridge = _pt(pts, 6)
        nose_tip = _pt(pts, 1)
        mouth_left = _pt(pts, 61)
        mouth_right = _pt(pts, 291)
        upper_lip = _pt(pts, 13)
        lower_lip = _pt(pts, 14)
        chin_center = _pt(pts, 152)
        left_brow_mid = _pt(pts, 105)
        right_brow_mid = _pt(pts, 334)
        left_brow_inner = _pt(pts, 70)
        left_brow_outer = _pt(pts, 107)
        right_brow_inner = _pt(pts, 336)
        right_brow_outer = _pt(pts, 300)

        face_w = _dist(left_face, right_face)
        face_h = _dist(forehead, chin)
        jaw_w = _dist(left_jaw, right_jaw)
        cheek_w = _dist(left_cheek, right_cheek)
        interocular = _dist(left_eye_inner, right_eye_inner)
        eye_left_h = _dist(left_eye_top, left_eye_bottom)
        eye_right_h = _dist(right_eye_top, right_eye_bottom)
        eye_size = None
        if eye_left_h is not None and eye_right_h is not None:
            eye_size = (eye_left_h + eye_right_h) * 0.5
        nose_w = _dist(nose_left, nose_right)
        nose_l = _dist(nose_bridge, nose_tip)
        mouth_w = _dist(mouth_left, mouth_right)
        lip_thickness = _dist(upper_lip, lower_lip)
        chin_len = _dist(lower_lip, chin_center)
        left_eye_center = None
        right_eye_center = None
        if left_eye_top is not None and left_eye_bottom is not None:
            left_eye_center = (left_eye_top + left_eye_bottom) * 0.5
        if right_eye_top is not None and right_eye_bottom is not None:
            right_eye_center = (right_eye_top + right_eye_bottom) * 0.5
        eye_center_dist = _dist(left_eye_center, right_eye_center)
        left_eye_w = _dist(left_eye_outer, left_eye_inner)
        right_eye_w = _dist(right_eye_outer, right_eye_inner)
        eye_size_full = None
        if eye_size is not None and left_eye_w is not None and right_eye_w is not None:
            eye_size_full = 0.5 * (eye_size + ((left_eye_w + right_eye_w) * 0.25))

        face_bbox_y_top = min(float(p[1]) for p in [p for p in [forehead, left_face, right_face] if p is not None]) if face_h else 0.0
        forehead_h = None
        if forehead is not None and left_eye_outer is not None and right_eye_outer is not None:
            brow_y = min(float(left_eye_outer[1]), float(right_eye_outer[1]))
            forehead_h = max(0.0, brow_y - float(face_bbox_y_top))
        eyebrow_h = None
        if left_eye_center is not None and right_eye_center is not None and left_brow_mid is not None and right_brow_mid is not None:
            eyebrow_h = (
                float(left_eye_center[1] - left_brow_mid[1]) + float(right_eye_center[1] - right_brow_mid[1])
            ) * 0.5
        eyebrow_angle = None
        if left_brow_inner is not None and left_brow_outer is not None and right_brow_inner is not None and right_brow_outer is not None:
            l_ang = np.arctan2(float(left_brow_outer[1] - left_brow_inner[1]), float(left_brow_outer[0] - left_brow_inner[0] + 1e-6))
            r_ang = np.arctan2(float(right_brow_inner[1] - right_brow_outer[1]), float(right_brow_inner[0] - right_brow_outer[0] + 1e-6))
            eyebrow_angle = float((l_ang + r_ang) * 0.5)

        params = {
            "faceWidthHeight": _safe_ratio(face_w, face_h, 0.78),
            "jawWidthRatio": _safe_ratio(jaw_w, face_w, 0.78),
            "cheekboneWidthRatio": _safe_ratio(cheek_w, face_w, 0.86),
            "foreheadHeightRatio": _safe_ratio(forehead_h, face_h, 0.30),
            "interocularRatio": _safe_ratio(interocular, face_w, 0.39),
            "eyeSizeRatio": _safe_ratio(eye_size, face_h, 0.16),
            "noseWidthRatio": _safe_ratio(nose_w, face_w, 0.24),
            "noseLengthRatio": _safe_ratio(nose_l, face_h, 0.27),
            "mouthWidthRatio": _safe_ratio(mouth_w, face_w, 0.36),
        }
        metrics = {
            "faceWidth": _safe_ratio(face_w, face_h, default_face_metrics()["faceWidth"]),
            "jawWidth": _safe_ratio(jaw_w, face_h, default_face_metrics()["jawWidth"]),
            "cheekboneWidth": _safe_ratio(cheek_w, face_h, default_face_metrics()["cheekboneWidth"]),
            "eyeDistance": _safe_ratio(eye_center_dist, face_h, default_face_metrics()["eyeDistance"]),
            "eyeSize": _safe_ratio(eye_size_full, face_h, default_face_metrics()["eyeSize"]),
            "eyebrowHeight": _safe_ratio(eyebrow_h, face_h, default_face_metrics()["eyebrowHeight"]),
            "eyebrowAngle": float(eyebrow_angle or default_face_metrics()["eyebrowAngle"]),
            "noseWidth": _safe_ratio(nose_w, face_h, default_face_metrics()["noseWidth"]),
            "noseLength": _safe_ratio(nose_l, face_h, default_face_metrics()["noseLength"]),
            "mouthWidth": _safe_ratio(mouth_w, face_h, default_face_metrics()["mouthWidth"]),
            "lipThickness": _safe_ratio(lip_thickness, face_h, default_face_metrics()["lipThickness"]),
            "chinLength": _safe_ratio(chin_len, face_h, default_face_metrics()["chinLength"]),
            "faceHeightRatio": _safe_ratio(face_h, face_w, default_face_metrics()["faceHeightRatio"]),
            "skullWidth": _safe_ratio(face_w, face_h, default_face_metrics()["skullWidth"]),
            "jawTaper": _safe_ratio(jaw_w, cheek_w, default_face_metrics()["jawTaper"]),
            "cheekboneProminence": _safe_ratio(cheek_w, jaw_w, default_face_metrics()["cheekboneProminence"]),
            "foreheadHeight": _safe_ratio(forehead_h, face_h, default_face_metrics()["foreheadHeight"]),
            "eyelidOpenness": _safe_ratio(eye_size, face_h, default_face_metrics()["eyelidOpenness"]),
        }

        # Clamp to sane ratio bounds for deterministic behavior.
        params = {
            "faceWidthHeight": _clamp(params["faceWidthHeight"], 0.55, 1.05),
            "jawWidthRatio": _clamp(params["jawWidthRatio"], 0.55, 1.05),
            "cheekboneWidthRatio": _clamp(params["cheekboneWidthRatio"], 0.65, 1.10),
            "foreheadHeightRatio": _clamp(params["foreheadHeightRatio"], 0.12, 0.50),
            "interocularRatio": _clamp(params["interocularRatio"], 0.20, 0.58),
            "eyeSizeRatio": _clamp(params["eyeSizeRatio"], 0.05, 0.26),
            "noseWidthRatio": _clamp(params["noseWidthRatio"], 0.10, 0.42),
            "noseLengthRatio": _clamp(params["noseLengthRatio"], 0.10, 0.52),
            "mouthWidthRatio": _clamp(params["mouthWidthRatio"], 0.18, 0.62),
        }
        metrics = {
            "faceWidth": _clamp(metrics["faceWidth"], 0.48, 1.06),
            "jawWidth": _clamp(metrics["jawWidth"], 0.35, 0.98),
            "cheekboneWidth": _clamp(metrics["cheekboneWidth"], 0.40, 1.02),
            "eyeDistance": _clamp(metrics["eyeDistance"], 0.16, 0.52),
            "eyeSize": _clamp(metrics["eyeSize"], 0.04, 0.24),
            "eyebrowHeight": _clamp(metrics["eyebrowHeight"], 0.01, 0.20),
            "eyebrowAngle": _clamp(metrics["eyebrowAngle"], -0.65, 0.65),
            "noseWidth": _clamp(metrics["noseWidth"], 0.08, 0.36),
            "noseLength": _clamp(metrics["noseLength"], 0.10, 0.50),
            "mouthWidth": _clamp(metrics["mouthWidth"], 0.12, 0.44),
            "lipThickness": _clamp(metrics["lipThickness"], 0.01, 0.16),
            "chinLength": _clamp(metrics["chinLength"], 0.06, 0.36),
            "faceHeightRatio": _clamp(metrics["faceHeightRatio"], 0.92, 1.92),
            "skullWidth": _clamp(metrics["skullWidth"], 0.48, 1.08),
            "jawTaper": _clamp(metrics["jawTaper"], 0.60, 1.35),
            "cheekboneProminence": _clamp(metrics["cheekboneProminence"], 0.80, 1.45),
            "foreheadHeight": _clamp(metrics["foreheadHeight"], 0.10, 0.55),
            "eyelidOpenness": _clamp(metrics["eyelidOpenness"], 0.02, 0.18),
        }
        avatar_morph_parameters = apply_face_morphs(metrics)

        # Stable skin tone sample from forehead + cheeks (avoid eyes/lips/hair zones).
        face_w_px = max(8.0, float(face_w or (w * 0.35)))
        face_h_px = max(8.0, float(face_h or (h * 0.45)))
        cx = float((left_face[0] + right_face[0]) * 0.5) if (left_face is not None and right_face is not None) else float(w * 0.5)
        cy = float((forehead[1] + chin[1]) * 0.5) if (forehead is not None and chin is not None) else float(h * 0.5)
        cheek_dx = face_w_px * 0.18
        cheek_w_px = face_w_px * 0.16
        cheek_h_px = face_h_px * 0.18
        forehead_w_px = face_w_px * 0.18
        forehead_h_px = face_h_px * 0.12

        left_cheek_patch = _crop(face_rgb, cx - cheek_dx - cheek_w_px, cy - cheek_h_px * 0.05, cx - cheek_dx, cy + cheek_h_px)
        right_cheek_patch = _crop(face_rgb, cx + cheek_dx, cy - cheek_h_px * 0.05, cx + cheek_dx + cheek_w_px, cy + cheek_h_px)
        forehead_patch = _crop(face_rgb, cx - forehead_w_px * 0.5, cy - face_h_px * 0.42, cx + forehead_w_px * 0.5, cy - face_h_px * 0.30)
        samples = [p for p in [left_cheek_patch, right_cheek_patch, forehead_patch] if p is not None and p.size > 0]
        color = None
        if samples:
            merged = np.vstack([s.reshape(-1, 3) for s in samples])
            # Light skin mask only to avoid hair/lips contamination (broad thresholds).
            ycrcb = cv2.cvtColor(merged.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
            Y, Cr, Cb = ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2]
            mask = (Y > 35) & (Cr >= 120) & (Cr <= 190) & (Cb >= 80) & (Cb <= 150)
            sel = merged[mask] if np.any(mask) else merged
            med = np.median(sel, axis=0)
            color = (int(med[0]), int(med[1]), int(med[2]))

        out["faceParams"] = params
        out["faceMetrics"] = metrics
        out["avatarMorphParameters"] = avatar_morph_parameters
        out["faceColorRgb8"] = list(color) if color is not None else None
        out["faceAnalysisSource"] = "facemesh"
        return out
    except Exception:
        return out


def _face_param_deltas(face_params: dict) -> dict[str, float]:
    # Empirical neutral baselines requested for morph mapping.
    ref = {
        "faceWidthHeight": 0.84,
        "jawWidthRatio": 0.83,
        "cheekboneWidthRatio": 1.00,
        "foreheadHeightRatio": 0.30,
        "interocularRatio": 0.21,
        "eyeSizeRatio": 0.16,
        "noseWidthRatio": 0.29,
        "noseLengthRatio": 0.23,
        "mouthWidthRatio": 0.34,
    }
    out = {}
    for k, rv in ref.items():
        v = float(face_params.get(k, rv))
        out[k] = (v - float(rv)) / max(abs(float(rv)), 1e-6)
    return out


def _head_geometry_names(scene: trimesh.Scene) -> list[str]:
    keywords = ("head", "face", "neck", "jaw")
    exclude = ("hair", "scalp", "brow", "eyebrow", "beard", "eye", "iris", "cornea", "sclera", "pupil")
    found: set[str] = set()
    try:
        for node in scene.graph.nodes:
            n = str(node).lower()
            if not any(k in n for k in keywords):
                continue
            if any(k in n for k in exclude):
                continue
            try:
                _, geom_name = scene.graph.get(node)
            except Exception:
                continue
            if geom_name in scene.geometry:
                found.add(str(geom_name))
    except Exception:
        pass
    for geom_name in scene.geometry.keys():
        low = str(geom_name).lower()
        if any(k in low for k in keywords) and not any(k in low for k in exclude):
            found.add(str(geom_name))
    if found:
        return sorted(found)
    return []


def apply_face_likeness_to_scene(scene: trimesh.Scene, face_params: dict, morph_params: Optional[dict] = None) -> dict:
    """
    Deterministic, subtle head-only deformation fallback (no blendshape dependency).
    """
    deltas = _face_param_deltas(face_params or {})
    m = dict(morph_params or {})
    head_names = _head_geometry_names(scene)
    if not head_names:
        return {"applied": False, "method": "none", "headTargetsMatched": [], "scalingApplied": {}}

    # Aggregate scale controls.
    face_width_delta = deltas["faceWidthHeight"] + 0.35 * deltas["jawWidthRatio"] + 0.35 * deltas["cheekboneWidthRatio"]
    face_length_delta = (-0.8 * deltas["faceWidthHeight"]) + 0.50 * deltas["foreheadHeightRatio"]
    eye_spacing_delta = deltas["interocularRatio"]
    nose_width_delta = deltas["noseWidthRatio"]
    nose_len_delta = deltas["noseLengthRatio"]
    mouth_width_delta = deltas["mouthWidthRatio"]
    skull_width = _clamp(float(m.get("skullWidth", 0.0)), -1.0, 1.0)
    jaw_taper = _clamp(float(m.get("jawTaper", 0.0)), -1.0, 1.0)
    cheek_prom = _clamp(float(m.get("cheekboneProminence", 0.0)), -1.0, 1.0)
    cheek_w = _clamp(float(m.get("cheekboneWidth", 0.0)), -1.0, 1.0)
    forehead_h = _clamp(float(m.get("foreheadHeight", 0.0)), -1.0, 1.0)
    eyelid_open = _clamp(float(m.get("eyelidOpenness", 0.0)), -1.0, 1.0)
    eye_size = _clamp(float(m.get("eyeSize", 0.0)), -1.0, 1.0)
    eyebrow_h = _clamp(float(m.get("eyebrowHeight", 0.0)), -1.0, 1.0)
    eyebrow_ang = _clamp(float(m.get("eyebrowAngle", 0.0)), -1.0, 1.0)

    scaling_applied = {
        "scaleXBase": round(1.0 + 0.16 * face_width_delta, 4),
        "scaleYBase": round(1.0 + 0.14 * face_length_delta, 4),
        "jawDelta": round(0.16 * deltas["jawWidthRatio"], 4),
        "cheekDelta": round(0.12 * deltas["cheekboneWidthRatio"], 4),
        "eyeSpacingDelta": round(0.08 * eye_spacing_delta, 4),
        "noseWidthDelta": round(0.07 * nose_width_delta, 4),
        "noseLengthDelta": round(0.08 * nose_len_delta, 4),
        "mouthWidthDelta": round(0.08 * mouth_width_delta, 4),
        "skullWidthDelta": round(0.07 * skull_width, 4),
        "jawTaperDelta": round(0.06 * jaw_taper, 4),
        "cheekboneProminenceDelta": round(0.06 * cheek_prom, 4),
        "cheekboneWidthDelta": round(0.05 * cheek_w, 4),
        "foreheadHeightDelta": round(0.05 * forehead_h, 4),
        "eyelidOpennessDelta": round(0.05 * eyelid_open, 4),
        "eyeSizeDelta": round(0.05 * eye_size, 4),
        "eyebrowHeightDelta": round(0.04 * eyebrow_h, 4),
        "eyebrowAngleDelta": round(0.03 * eyebrow_ang, 4),
    }

    touched = 0
    for name in head_names:
        g = scene.geometry.get(name)
        if g is None or not hasattr(g, "vertices") or len(g.vertices) == 0:
            continue
        try:
            v = np.asarray(g.vertices, dtype=np.float64, copy=True)
            b = np.asarray(g.bounds, dtype=np.float64).reshape(2, 3)
            cx = float((b[0, 0] + b[1, 0]) * 0.5)
            cy_min = float(b[0, 1])
            cy_max = float(b[1, 1])
            cz = float((b[0, 2] + b[1, 2]) * 0.5)
            span_y = max(1e-6, cy_max - cy_min)
            t = (v[:, 1] - cy_min) / span_y

            jaw_band = np.exp(-0.5 * ((t - 0.20) / 0.12) ** 2)
            cheek_band = np.exp(-0.5 * ((t - 0.46) / 0.10) ** 2)
            eye_band = np.exp(-0.5 * ((t - 0.62) / 0.08) ** 2)
            nose_band = np.exp(-0.5 * ((t - 0.52) / 0.08) ** 2)
            mouth_band = np.exp(-0.5 * ((t - 0.36) / 0.06) ** 2)
            forehead_band = np.exp(-0.5 * ((t - 0.82) / 0.10) ** 2)
            brow_band = np.exp(-0.5 * ((t - 0.70) / 0.06) ** 2)

            sx = (
                scaling_applied["scaleXBase"]
                + scaling_applied["jawDelta"] * jaw_band
                + scaling_applied["cheekDelta"] * cheek_band
                + scaling_applied["eyeSpacingDelta"] * eye_band
                + scaling_applied["noseWidthDelta"] * nose_band * 0.7
                + scaling_applied["mouthWidthDelta"] * mouth_band
                + scaling_applied["skullWidthDelta"] * forehead_band
                + scaling_applied["cheekboneProminenceDelta"] * cheek_band
                + scaling_applied["cheekboneWidthDelta"] * cheek_band * 0.8
                + scaling_applied["eyeSizeDelta"] * eye_band * 0.4
                - scaling_applied["jawTaperDelta"] * jaw_band * 0.7
            )
            sx = np.clip(sx, 0.82, 1.22)
            sy = np.clip(
                scaling_applied["scaleYBase"]
                + scaling_applied["noseLengthDelta"] * nose_band * 0.5
                + scaling_applied["foreheadHeightDelta"] * forehead_band
                + scaling_applied["eyebrowHeightDelta"] * brow_band * 0.4
                - scaling_applied["eyelidOpennessDelta"] * eye_band * 0.2,
                0.86,
                1.20,
            )

            v[:, 0] = cx + (v[:, 0] - cx) * sx
            v[:, 1] = cy_min + (v[:, 1] - cy_min) * sy
            # Keep Z subtle to avoid profile breakage.
            sz = np.clip(
                1.0
                + 0.05 * face_width_delta
                + 0.02 * scaling_applied["cheekboneProminenceDelta"] * cheek_band
                + 0.02 * scaling_applied["eyebrowAngleDelta"] * brow_band,
                0.92,
                1.10,
            )
            v[:, 2] = cz + (v[:, 2] - cz) * sz
            g.vertices = v
            touched += 1
        except Exception:
            continue

    return {
        "applied": bool(touched > 0),
        "method": "scale_fallback",
        "headTargetsMatched": head_names,
        "headGeometryTouched": int(touched),
        "scalingApplied": scaling_applied,
        "deltas": {k: round(float(v), 4) for k, v in deltas.items()},
    }


def _head_mesh_indices_from_gltf(gltf) -> set[int]:
    idxs: set[int] = set()
    try:
        for node in gltf.nodes or []:
            mesh_idx = getattr(node, "mesh", None)
            if mesh_idx is None:
                continue
            node_name = str(getattr(node, "name", "") or "").lower()
            mesh_name = ""
            try:
                mesh_name = str(getattr(gltf.meshes[mesh_idx], "name", "") or "").lower()
            except Exception:
                mesh_name = ""
            if any(k in node_name or k in mesh_name for k in ("head", "face", "neck", "jaw", "brow")):
                idxs.add(int(mesh_idx))
    except Exception:
        return set()
    return idxs


def _extract_target_names(mesh) -> list[str]:
    names = []
    extras = getattr(mesh, "extras", None)
    if isinstance(extras, dict):
        target_names = extras.get("targetNames")
        if isinstance(target_names, list):
            names = [str(x) for x in target_names]
    prims = getattr(mesh, "primitives", None) or []
    count = 0
    try:
        if prims and getattr(prims[0], "targets", None) is not None:
            count = len(prims[0].targets)
    except Exception:
        count = 0
    if not names and count > 0:
        names = [f"target_{i}" for i in range(count)]
    if count > len(names):
        names.extend([f"target_{i}" for i in range(len(names), count)])
    return names


def inspect_head_morph_targets(glb_path: str) -> dict:
    try:
        from pygltflib import GLTF2
    except Exception:
        return {
            "hasHeadMorphTargets": False,
            "morphTargetNamesByMesh": {},
            "headMeshes": [],
            "morphTargetsFoundCount": 0,
            "method": "morph_unavailable",
        }

    p = Path(glb_path)
    if not p.exists():
        return {
            "hasHeadMorphTargets": False,
            "morphTargetNamesByMesh": {},
            "headMeshes": [],
            "morphTargetsFoundCount": 0,
            "method": "morph_unavailable",
        }

    try:
        gltf = GLTF2().load(str(p))
        mesh_indices = _head_mesh_indices_from_gltf(gltf)
        if not mesh_indices:
            # Fallback: pick meshes that actually expose morph targets.
            for idx, mesh in enumerate(gltf.meshes or []):
                targets = _extract_target_names(mesh)
                if targets:
                    mesh_indices.add(int(idx))
        head_meshes = []
        names_by_mesh: dict[str, list[str]] = {}
        total = 0
        for idx in sorted(mesh_indices):
            if idx >= len(gltf.meshes):
                continue
            mesh = gltf.meshes[idx]
            mname = str(getattr(mesh, "name", None) or f"mesh_{idx}")
            targets = _extract_target_names(mesh)
            total += len(targets)
            names_by_mesh[mname] = targets
            head_meshes.append({"meshIndex": int(idx), "meshName": mname, "hasMorphTargets": bool(targets), "morphTargetCount": len(targets)})
        has = total > 0
        return {
            "hasHeadMorphTargets": bool(has),
            "morphTargetNamesByMesh": names_by_mesh,
            "headMeshes": head_meshes,
            "morphTargetsFoundCount": int(total),
            "method": "morph_detected" if has else "morph_missing",
        }
    except Exception as e:
        logger.warning("inspect morph targets failed: %s", e)
        return {
            "hasHeadMorphTargets": False,
            "morphTargetNamesByMesh": {},
            "headMeshes": [],
            "morphTargetsFoundCount": 0,
            "method": "morph_failed",
        }


def inspect_exported_glb_structure(glb_path: str) -> dict:
    """
    Definitive exported-GLB inspection:
    - all meshes with node/mesh/material names
    - morph target count + names
    - head/hair/eye candidate lists
    """
    out = {
        "meshInspection": [],
        "headMeshNames": [],
        "hairMeshNames": [],
        "eyeMeshNames": [],
        "morphTargetsByMesh": {},
        "hasHeadMorphTargets": False,
    }
    try:
        from pygltflib import GLTF2
    except Exception:
        return out
    p = Path(glb_path)
    if not p.exists():
        return out
    try:
        gltf = GLTF2().load(str(p))
        mats = gltf.materials or []
        mesh_nodes: dict[int, list[str]] = {}
        for node in gltf.nodes or []:
            mi = getattr(node, "mesh", None)
            if mi is None:
                continue
            mesh_nodes.setdefault(int(mi), []).append(str(getattr(node, "name", None) or f"node_{len(mesh_nodes)}"))

        head, hair, eye = set(), set(), set()
        morph_map: dict[str, list[str]] = {}
        mesh_rows = []
        for mi, mesh in enumerate(gltf.meshes or []):
            mesh_name = str(getattr(mesh, "name", None) or f"mesh_{mi}")
            node_names = mesh_nodes.get(mi, [])
            prims = getattr(mesh, "primitives", None) or []
            mat_names = []
            for prim in prims:
                m_idx = getattr(prim, "material", None)
                if isinstance(m_idx, int) and 0 <= m_idx < len(mats):
                    mat_names.append(str(getattr(mats[m_idx], "name", None) or f"material_{m_idx}"))
            mat_names = sorted(set(mat_names))
            targets = _extract_target_names(mesh)
            morph_map[mesh_name] = list(targets)
            all_names_low = " ".join([mesh_name] + node_names + mat_names).lower()
            if any(k in all_names_low for k in HEAD_KWS) and not any(k in all_names_low for k in HAIR_KWS + EYE_KWS):
                head.add(mesh_name)
            if any(k in all_names_low for k in HAIR_KWS):
                hair.add(mesh_name)
            if any(k in all_names_low for k in EYE_KWS):
                eye.add(mesh_name)
            mesh_rows.append(
                {
                    "meshName": mesh_name,
                    "nodeNames": node_names,
                    "materialNames": mat_names,
                    "morphTargetCount": int(len(targets)),
                    "morphTargetNames": list(targets),
                }
            )
        out["meshInspection"] = mesh_rows
        out["headMeshNames"] = sorted(head)
        out["hairMeshNames"] = sorted(hair)
        out["eyeMeshNames"] = sorted(eye)
        out["morphTargetsByMesh"] = morph_map
        out["hasHeadMorphTargets"] = any(len(morph_map.get(m, [])) > 0 for m in out["headMeshNames"])
        return out
    except Exception:
        return out


def apply_face_likeness_to_glb_morphs(glb_path: str, face_params: dict, morph_params: Optional[dict] = None) -> dict:
    try:
        from pygltflib import GLTF2
    except Exception:
        return {"applied": False, "method": "morph_unavailable", "morphTargetsApplied": [], "morphTargetNamesByMesh": {}, "hasHeadMorphTargets": False}

    p = Path(glb_path)
    if not p.exists():
        return {"applied": False, "method": "morph_unavailable", "morphTargetsApplied": [], "morphTargetNamesByMesh": {}, "hasHeadMorphTargets": False}

    inspect = inspect_head_morph_targets(glb_path)
    if not inspect.get("hasHeadMorphTargets"):
        return {
            "applied": False,
            "method": "morph_missing",
            "morphTargetsApplied": [],
            "morphTargetNamesByMesh": inspect.get("morphTargetNamesByMesh", {}),
            "hasHeadMorphTargets": False,
        }

    deltas = _face_param_deltas(face_params or {})
    semantic_raw = {
        "jawWidth": _clamp(float(deltas.get("jawWidthRatio", 0.0)) * 1.15, -1.0, 1.0),
        "jawHeight": _clamp(float(deltas.get("faceWidthHeight", 0.0)) * -0.95, -1.0, 1.0),
        "cheekboneWidth": _clamp(float(deltas.get("cheekboneWidthRatio", 0.0)) * 1.10, -1.0, 1.0),
        "noseWidth": _clamp(float(deltas.get("noseWidthRatio", 0.0)) * 1.25, -1.0, 1.0),
        "noseLength": _clamp(float(deltas.get("noseLengthRatio", 0.0)) * 1.20, -1.0, 1.0),
        "eyeSize": _clamp(float(deltas.get("eyeSizeRatio", 0.0)) * 1.10, -1.0, 1.0),
        "faceWidth": _clamp(float(deltas.get("faceWidthHeight", 0.0)) * 1.05, -1.0, 1.0),
        "chinSize": _clamp((float(deltas.get("mouthWidthRatio", 0.0)) * 0.5) + (float(deltas.get("jawWidthRatio", 0.0)) * 0.5), -1.0, 1.0),
    }
    mp = dict(morph_params or {})
    semantic_raw["jawWidth"] = _clamp(0.55 * semantic_raw["jawWidth"] + 0.45 * float(mp.get("jawWidth", 0.0)), -1.0, 1.0)
    semantic_raw["noseWidth"] = _clamp(0.60 * semantic_raw["noseWidth"] + 0.40 * float(mp.get("noseWidth", 0.0)), -1.0, 1.0)
    semantic_raw["noseLength"] = _clamp(0.60 * semantic_raw["noseLength"] + 0.40 * float(mp.get("noseLength", 0.0)), -1.0, 1.0)
    semantic_raw["chinSize"] = _clamp(0.55 * semantic_raw["chinSize"] + 0.45 * float(mp.get("chinHeight", 0.0)), -1.0, 1.0)
    semantic_raw["faceWidth"] = _clamp(
        0.55 * semantic_raw["faceWidth"] + 0.20 * float(mp.get("skullWidth", 0.0)) + 0.25 * float(mp.get("faceHeight", 0.0)),
        -1.0,
        1.0,
    )
    semantic_raw["eyeSize"] = _clamp(
        0.55 * semantic_raw["eyeSize"]
        + 0.25 * float(mp.get("eyelidOpenness", 0.0))
        + 0.20 * float(mp.get("eyeSize", 0.0)),
        -1.0,
        1.0,
    )
    semantic_raw["cheekboneWidth"] = _clamp(
        0.70 * semantic_raw["cheekboneWidth"] + 0.30 * float(mp.get("cheekboneWidth", 0.0)),
        -1.0,
        1.0,
    )
    semantic_weights = {k: round(_weight_smooth(v), 4) for k, v in semantic_raw.items()}
    alias_to_semantic = {
        "jawwidth": "jawWidth",
        "jaww": "jawWidth",
        "jawwide": "jawWidth",
        "jawheight": "jawHeight",
        "jawh": "jawHeight",
        "cheekbonewidth": "cheekboneWidth",
        "cheekwidth": "cheekboneWidth",
        "cheekbones": "cheekboneWidth",
        "nosewidth": "noseWidth",
        "noselength": "noseLength",
        "eyesize": "eyeSize",
        "eyescale": "eyeSize",
        "facewidth": "faceWidth",
        "headwidth": "faceWidth",
        "chinsize": "chinSize",
        "chinwidth": "chinSize",
        "eyespacing": "eyeSize",
        "eyelid": "eyeSize",
        "brow": "cheekboneWidth",
        "forehead": "faceWidth",
    }

    try:
        gltf = GLTF2().load(str(p))
        applied = []
        changed = False
        if not gltf.meshes:
            return {"applied": False, "method": "morph_missing", "morphTargetsApplied": [], "morphTargetNamesByMesh": {}, "hasHeadMorphTargets": False}

        head_mesh_idxs = _head_mesh_indices_from_gltf(gltf)
        for node in (gltf.nodes or []):
            mesh_idx = getattr(node, "mesh", None)
            if mesh_idx is None or mesh_idx >= len(gltf.meshes) or int(mesh_idx) not in head_mesh_idxs:
                continue
            mesh = gltf.meshes[mesh_idx]
            target_names = _extract_target_names(mesh)
            if not target_names:
                continue
            if node.weights is None:
                node.weights = [0.0] * len(target_names)
            while len(node.weights) < len(target_names):
                node.weights.append(0.0)

            for i, tn in enumerate(target_names):
                tname = _norm_name(tn)
                semantic = None
                for alias, semantic_name in alias_to_semantic.items():
                    if alias in tname:
                        semantic = semantic_name
                        break
                if semantic is None:
                    continue
                weight = float(semantic_weights.get(semantic, 0.5))
                node.weights[i] = float(weight)
                changed = True
                applied.append(
                    {
                        "node": getattr(node, "name", None),
                        "mesh": getattr(mesh, "name", None),
                        "targetName": tn,
                        "semantic": semantic,
                        "weight": round(float(weight), 4),
                    }
                )

        if changed:
            gltf.save(str(p))
            logger.info("morph weights applied: %s", semantic_weights)
            return {
                "applied": True,
                "method": "morph",
                "morphTargetsApplied": applied,
                "morphTargetNamesByMesh": inspect.get("morphTargetNamesByMesh", {}),
                "morphWeights": semantic_weights,
                "hasHeadMorphTargets": True,
            }
        return {
            "applied": False,
            "method": "morph_missing",
            "morphTargetsApplied": [],
            "morphTargetNamesByMesh": inspect.get("morphTargetNamesByMesh", {}),
            "morphWeights": semantic_weights,
            "hasHeadMorphTargets": True,
        }
    except Exception as e:
        logger.warning("face morph apply failed: %s", e)
        return {
            "applied": False,
            "method": "morph_failed",
            "morphTargetsApplied": [],
            "morphTargetNamesByMesh": inspect.get("morphTargetNamesByMesh", {}),
            "morphWeights": {},
            "hasHeadMorphTargets": bool(inspect.get("hasHeadMorphTargets")),
        }


def _feature_poly(points: np.ndarray, idxs: list[int]) -> Optional[np.ndarray]:
    out = []
    for i in idxs:
        p = _pt(points, i)
        if p is not None:
            out.append(p)
    if len(out) < 3:
        return None
    return np.asarray(out, dtype=np.float32)


def _normalize_face_lighting(face_rgb: np.ndarray) -> np.ndarray:
    x = face_rgb.astype(np.uint8)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    balanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    means = np.mean(balanced.reshape(-1, 3), axis=0)
    gray = float(np.mean(means))
    scale = np.clip(gray / np.maximum(means, 1e-6), 0.85, 1.15)
    out = np.clip(balanced.astype(np.float32) * scale.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    return out


def generate_face_overlay_rgba(face_rgb: np.ndarray, landmarks_xy_px: Optional[np.ndarray], size: int = 512) -> tuple[Optional[np.ndarray], dict]:
    meta = {
        "beardDetected": False,
        "beardOpacity": 0.0,
        "faceBoundingBoxPx": None,
        "alignmentApplied": False,
    }
    if face_rgb is None or face_rgb.size == 0:
        return None, meta
    if landmarks_xy_px is None or len(landmarks_xy_px) < 100:
        return None, meta
    pts = np.asarray(landmarks_xy_px, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))
    meta["faceBoundingBoxPx"] = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    pad_x = (x2 - x1) * 0.18
    pad_y = (y2 - y1) * 0.24
    # Landmark-based face alignment to canonical frame.
    try:
        l_eye = _pt(pts, 33)
        r_eye = _pt(pts, 263)
        mouth = _pt(pts, 13) or _pt(pts, 14) or _pt(pts, 0)
        if l_eye is not None and r_eye is not None and mouth is not None:
            src_tri = np.float32([l_eye, r_eye, mouth])
            dst_tri = np.float32(
                [
                    [size * 0.34, size * 0.36],
                    [size * 0.66, size * 0.36],
                    [size * 0.50, size * 0.67],
                ]
            )
            M = cv2.getAffineTransform(src_tri, dst_tri)
            aligned = cv2.warpAffine(face_rgb, M, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
            ov_rgb = _normalize_face_lighting(aligned)
            meta["alignmentApplied"] = True
        else:
            crop = _crop(face_rgb, x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
            if crop is None:
                return None, meta
            crop = _normalize_face_lighting(crop)
            ov_rgb = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    except Exception:
        crop = _crop(face_rgb, x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
        if crop is None:
            return None, meta
        crop = _normalize_face_lighting(crop)
        ov_rgb = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    alpha = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(alpha, (size // 2, int(size * 0.54)), (int(size * 0.34), int(size * 0.44)), 0, 0, 360, 150, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(2, int(size * 0.03)))

    # Exclusion masks (eyes/eyebrows/lips) - reduce opacity strongly there.
    def to_crop_space(p: np.ndarray) -> np.ndarray:
        px = (p[0] - (x1 - pad_x)) / max(1e-6, (x2 - x1 + 2 * pad_x))
        py = (p[1] - (y1 - pad_y)) / max(1e-6, (y2 - y1 + 2 * pad_y))
        return np.asarray([px * size, py * size], dtype=np.float32)

    pts2 = np.asarray([to_crop_space(p) for p in pts], dtype=np.float32)
    eye_left = _feature_poly(pts2, [33, 160, 159, 158, 133, 153, 145, 144])
    eye_right = _feature_poly(pts2, [362, 385, 386, 387, 263, 373, 374, 380])
    brow_left = _feature_poly(pts2, [70, 63, 105, 66, 107, 55, 65, 52])
    brow_right = _feature_poly(pts2, [300, 293, 334, 296, 336, 285, 295, 282])
    lips = _feature_poly(pts2, [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78])
    for poly in [eye_left, eye_right, brow_left, brow_right, lips]:
        if poly is not None:
            cv2.fillPoly(alpha, [poly.astype(np.int32)], 18)

    # Optional subtle beard darkening (lower face only, safe and deterministic).
    lower = ov_rgb[int(size * 0.56):, :, :]
    if lower.size > 0:
        gray = cv2.cvtColor(lower, cv2.COLOR_RGB2GRAY)
        if float(np.mean(gray)) < 105.0:
            beard_mask = np.zeros((size, size), dtype=np.uint8)
            cv2.ellipse(beard_mask, (size // 2, int(size * 0.72)), (int(size * 0.22), int(size * 0.17)), 0, 0, 360, 1, -1)
            beard_mask = cv2.GaussianBlur(beard_mask.astype(np.float32), (0, 0), sigmaX=max(2, int(size * 0.02)))
            beard_opacity = 0.26
            dark = (1.0 - beard_opacity * beard_mask[..., None])
            ov_rgb = np.clip(ov_rgb.astype(np.float32) * dark, 0, 255).astype(np.uint8)
            meta["beardDetected"] = True
            meta["beardOpacity"] = float(beard_opacity)

    rgba = np.dstack([ov_rgb, alpha]).astype(np.uint8)
    return rgba, meta


def _head_materials(scene: trimesh.Scene) -> list[object]:
    mats = []
    seen = set()
    for gname in _head_geometry_names(scene):
        g = scene.geometry.get(gname)
        if g is None:
            continue
        mat = getattr(getattr(g, "visual", None), "material", None)
        if mat is None:
            continue
        key = id(mat)
        if key in seen:
            continue
        seen.add(key)
        mats.append(mat)
    return mats


def apply_texture_overlay_fallback(
    scene: trimesh.Scene,
    face_rgb: np.ndarray,
    landmarks_xy_px: Optional[np.ndarray],
    request_id: Optional[str] = None,
) -> dict:
    overlay, meta = generate_face_overlay_rgba(face_rgb, landmarks_xy_px, size=512)
    if overlay is None and DEBUG_FORCE_TEXTURE and face_rgb is not None and face_rgb.size > 0:
        forced = cv2.resize(face_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
        alpha = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(alpha, (256, 268), (170, 218), 0, 0, 360, 170, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=14)
        overlay = np.dstack([forced, alpha]).astype(np.uint8)
    if overlay is None:
        return {
            "applied": False,
            "method": "none",
            "materials": [],
            "overlayOpacity": 0.35,
            "beardDetected": False,
            "beardOpacity": 0.0,
            "textureResolution": None,
            "headMaterialName": None,
        }

    materials = _head_materials(scene)
    if not materials:
        return {
            "applied": False,
            "method": "none",
            "materials": [],
            "overlayOpacity": 0.35,
            "beardDetected": False,
            "beardOpacity": 0.0,
            "textureResolution": None,
            "headMaterialName": None,
        }

    updated = []
    material_apply_failures = 0
    ov_rgb = overlay[..., :3].astype(np.float32)
    ov_a = (overlay[..., 3].astype(np.float32) / 255.0) * 0.42  # subtle blend
    for mat in materials:
        try:
            base_img = getattr(mat, "image", None)
            if isinstance(base_img, Image.Image):
                base_np = np.array(base_img.convert("RGB"))
            elif isinstance(base_img, np.ndarray):
                base_np = base_img
            else:
                base_np = np.full((1024, 1024, 3), 178, dtype=np.uint8)
            h, w = base_np.shape[:2]
            # UV-projection style blend: place aligned face into center UV island zone.
            canvas_rgb = base_np.astype(np.float32).copy()
            canvas_a = np.zeros((h, w), dtype=np.float32)
            uv_x1, uv_x2 = int(round(w * 0.30)), int(round(w * 0.70))
            uv_y1, uv_y2 = int(round(h * 0.10)), int(round(h * 0.64))
            region_w = max(4, uv_x2 - uv_x1)
            region_h = max(4, uv_y2 - uv_y1)
            ov_r = cv2.resize(ov_rgb.astype(np.uint8), (region_w, region_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            ov_ar = cv2.resize(ov_a.astype(np.float32), (region_w, region_h), interpolation=cv2.INTER_CUBIC)
            # Preserve neck transition by matching overlay chroma to underlying UV neck/face area.
            base_region = base_np[uv_y1:uv_y2, uv_x1:uv_x2].astype(np.float32)
            if base_region.size > 0 and ov_r.size > 0:
                ov_mean = np.mean(ov_r.reshape(-1, 3), axis=0)
                base_mean = np.mean(base_region.reshape(-1, 3), axis=0)
                ratio = np.clip(base_mean / np.maximum(ov_mean, 1e-6), 0.78, 1.22)
                ov_r = np.clip(ov_r * ratio.reshape(1, 1, 3), 0, 255)
            feather = np.zeros((region_h, region_w), dtype=np.float32)
            cv2.ellipse(feather, (region_w // 2, int(region_h * 0.52)), (int(region_w * 0.46), int(region_h * 0.48)), 0, 0, 360, 1.0, -1)
            feather = cv2.GaussianBlur(feather, (0, 0), sigmaX=max(2, int(region_w * 0.03)))
            ov_ar = np.clip(ov_ar * feather, 0.0, 0.62)
            canvas_rgb[uv_y1:uv_y2, uv_x1:uv_x2] = ov_r
            canvas_a[uv_y1:uv_y2, uv_x1:uv_x2] = ov_ar
            out = np.clip(base_np.astype(np.float32) * (1.0 - canvas_a[..., None]) + canvas_rgb * canvas_a[..., None], 0, 255).astype(np.uint8)
            out_img = Image.fromarray(out, mode="RGB")
            applied = False
            # Some trimesh material types accept `image`, others reject `baseColorTexture` assignment.
            try:
                if hasattr(mat, "image"):
                    mat.image = out_img
                    applied = True
            except Exception:
                pass
            try:
                if hasattr(mat, "baseColorTexture"):
                    mat.baseColorTexture = out_img
                    applied = True
            except Exception:
                pass
            if applied:
                updated.append(str(getattr(mat, "name", None) or type(mat).__name__))
            else:
                material_apply_failures += 1
        except Exception:
            material_apply_failures += 1
            continue
    logger.info("texture overlay fallback applied request_id=%s materials=%s", request_id, updated)
    return {
        "applied": bool(updated),
        "method": "texture_overlay" if updated else "none",
        "materials": updated,
        "overlayOpacity": 0.35,
        "beardDetected": bool(meta.get("beardDetected")),
        "beardOpacity": float(meta.get("beardOpacity") or 0.0),
        "textureResolution": [int(overlay.shape[1]), int(overlay.shape[0])],
        "headMaterialName": updated[0] if updated else None,
        "faceBoundingBoxPx": meta.get("faceBoundingBoxPx"),
        "alignmentApplied": bool(meta.get("alignmentApplied")),
        "materialApplyFailures": int(material_apply_failures),
    }
