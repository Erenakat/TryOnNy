"""
Avatar pipeline: imported mannequin GLB only. No procedural/primitive geometry.
- Body: pre-made humanoid GLB (CesiumMan), neutral pose, <80k tris.
- Face: texture applied to head mesh.
- Materials: PBR (baseColor, roughness 0.5), GLB export.
"""
import hashlib
import logging
import os
import base64
import mimetypes
import binascii
import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh
from trimesh.visual.material import SimpleMaterial
from PIL import Image

try:
    from trimesh.visual.material import PBRMaterial
except Exception:
    PBRMaterial = None

from models_download import get_face_model_path
from errors import AppError, classify_pose_exception, sanitize_for_json
from pose_service import pose_service
from gltf_debug import gltf_debug_morphs
from face_likeness import (
    analyze_face_landmarks,
    apply_face_morphs,
    apply_face_likeness_to_scene,
    apply_face_likeness_to_glb_morphs,
    apply_texture_overlay_fallback,
    default_face_params,
    default_face_metrics,
    inspect_exported_glb_structure,
    inspect_head_morph_targets,
    validate_likeness_and_refine,
)

# Mobile-friendly: max texture size, poly budget (~50–80k tris)
TEX_SIZE = 1024

# 1x1 white texture: when used as baseColorTexture, baseColorFactor * white = baseColorFactor (skin color shows).
_WHITE_1X1: Optional[Image.Image] = None
_BLACK_1X1: Optional[Image.Image] = None


def _get_white_1x1_texture() -> Image.Image:
    global _WHITE_1X1
    if _WHITE_1X1 is None:
        _WHITE_1X1 = Image.new("RGB", (1, 1), (255, 255, 255))
    return _WHITE_1X1


def _get_black_1x1_texture() -> Image.Image:
    global _BLACK_1X1
    if _BLACK_1X1 is None:
        _BLACK_1X1 = Image.new("RGB", (1, 1), (0, 0, 0))
    return _BLACK_1X1

# Proportion scaling defaults (used by body fitting)
BODY_SCALE_DEFAULTS = {
    "heightScale": 1.0,
    "shoulderWidthScale": 1.0,
    "chestScale": 1.0,
    "waistScale": 1.0,
    "hipScale": 1.0,
    "legLengthScale": 1.0,
    "torsoLengthScale": 1.0,
}

# Mannequin GLB asset: auto-downloaded by download_mannequin.py
BASE_MANNEQUIN_DIR = Path(__file__).resolve().parent / "static" / "avatars"
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
LIGHTING_MODE = "studio_env_v2"
FIXED_HUMAN_SKIN_SRGB = (0.72, 0.56, 0.47)
USE_LOCKED_BASE_AVATAR = os.getenv("USE_LOCKED_BASE_AVATAR", "0").strip().lower() in {"1", "true", "yes", "on"}

# Reference body ratios for neutral mannequin fitting (front image space)
REF_SHOULDER_RATIO = 0.26  # shoulder_width / body_height
REF_HIP_RATIO = 0.22       # hip_width / body_height
REF_LEG_RATIO = 0.50       # hip_to_ankle / body_height
REF_TORSO_RATIO = 0.30     # shoulder_to_hip / body_height
REF_SLENDER = 4.0          # body_height / shoulder_width
REF_TOTAL_HEIGHT_RATIO = 0.78  # body_height / image_height
POSE_MIN_CONFIDENCE = 0.35
ENABLE_BODY_ANALYSIS = os.getenv("ENABLE_BODY_ANALYSIS", "1").strip().lower() in {"1", "true", "yes", "on"}
MIN_BODY_FRONT_WIDTH_PX = 160
MIN_BODY_FRONT_HEIGHT_PX = 220

# TEMP DEBUG: bypass pose detection to verify mesh scaling path.
USE_HARDCODED_BODY_MEASUREMENTS = False
HARDCODED_BODY_MEASUREMENTS = {
    "shoulderWidthPx": 250.0,
    "hipWidthPx": 200.0,
    "legLengthPx": 500.0,
    "torsoLengthPx": 300.0,
    "totalHeightPx": 800.0,
}

# Debug deformation: make personalization visibly obvious while testing.
# Default OFF so production output does not always look "debug/mannequin".
DEBUG_VISUAL_DEFORM = os.getenv("DEBUG_VISUAL_DEFORM", "0") == "1"
EXTREME_DEBUG_DEFORM = False
DEBUG_FORCE_SKIN_COLOR = os.getenv("DEBUG_FORCE_SKIN_COLOR", "").strip()


def _srgb_to_linear_channel(v: float) -> float:
    # v in [0,1]
    if v <= 0.04045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


def _rgb8_to_linear_factor(rgb: tuple[int, int, int]) -> list[float]:
    r = _srgb_to_linear_channel(float(rgb[0]) / 255.0)
    g = _srgb_to_linear_channel(float(rgb[1]) / 255.0)
    b = _srgb_to_linear_channel(float(rgb[2]) / 255.0)
    return [float(r), float(g), float(b), 1.0]


def _material_name(mat) -> str:
    try:
        n = getattr(mat, "name", None)
        return str(n) if n is not None else type(mat).__name__
    except Exception:
        return type(mat).__name__


def _material_matches_keywords(mat, keywords: tuple[str, ...]) -> bool:
    name = _material_name(mat).lower()
    return any(k in name for k in keywords)


def _geometry_material(geom) -> Optional[object]:
    try:
        vis = getattr(geom, "visual", None)
        mat = getattr(vis, "material", None) if vis is not None else None
        return mat
    except Exception:
        return None


def _material_usage_weights(scene: trimesh.Scene) -> dict[int, dict]:
    """
    Compute per-material usage weights based on geometry faces/vertices.
    Returns {id(mat): {"mat": mat, "name": str, "weight": float}}
    """
    out: dict[int, dict] = {}
    for g in scene.geometry.values():
        mat = _geometry_material(g)
        if mat is None:
            continue
        try:
            weight = float(len(getattr(g, "faces", []))) if hasattr(g, "faces") and getattr(g, "faces", None) is not None else float(len(getattr(g, "vertices", [])))
        except Exception:
            weight = 1.0
        key = id(mat)
        if key not in out:
            out[key] = {"mat": mat, "name": _material_name(mat), "weight": 0.0}
        out[key]["weight"] += weight
    return out


def _select_materials(scene: trimesh.Scene, *, keywords: tuple[str, ...], fallback_top_n: int = 1) -> list[object]:
    weights = _material_usage_weights(scene)
    mats = [v["mat"] for v in weights.values() if _material_matches_keywords(v["mat"], keywords)]
    if mats:
        return mats
    # Fallback: use top N most-used materials (usually body)
    ranked = sorted(weights.values(), key=lambda x: float(x.get("weight", 0.0)), reverse=True)
    return [x["mat"] for x in ranked[: max(1, int(fallback_top_n))]]


def _select_head_materials(scene: trimesh.Scene) -> list[object]:
    # 1) Prefer scene graph nodes that look like head/face
    head_keywords = ("head", "face", "neck")
    mats = []
    try:
        for node in scene.graph.nodes:
            node_name = str(node).lower()
            if not any(k in node_name for k in head_keywords):
                continue
            try:
                _, geom_name = scene.graph.get(node)
            except Exception:
                continue
            if geom_name in scene.geometry:
                mat = _geometry_material(scene.geometry[geom_name])
                if mat is not None:
                    mats.append(mat)
    except Exception:
        mats = []
    # 2) Fallback by material names
    if not mats:
        mats = _select_materials(scene, keywords=head_keywords, fallback_top_n=0)
    # Deduplicate by object identity
    uniq = []
    seen = set()
    for m in mats:
        if id(m) in seen:
            continue
        seen.add(id(m))
        uniq.append(m)
    return uniq


def _select_body_materials(scene: trimesh.Scene) -> list[object]:
    """
    Select materials likely used by the body surface.
    Prefer materials referenced by torso/limb-like nodes in the scene graph,
    then fall back to keyword/name matching, then to most-used materials.
    """
    body_node_keywords = ("torso", "spine", "pelvis", "hip", "thigh", "calf", "leg", "knee", "arm", "forearm", "hand", "shoulder")
    mats = []
    try:
        for node in scene.graph.nodes:
            node_name = str(node).lower()
            if not any(k in node_name for k in body_node_keywords):
                continue
            try:
                _, geom_name = scene.graph.get(node)
            except Exception:
                continue
            if geom_name in scene.geometry:
                mat = _geometry_material(scene.geometry[geom_name])
                if mat is not None:
                    mats.append(mat)
    except Exception:
        mats = []

    # Fallback by material name keywords
    if not mats:
        skin_keywords = ("skin", "body", "torso", "arm", "leg", "head", "face")
        mats = _select_materials(scene, keywords=skin_keywords, fallback_top_n=2)

    uniq = []
    seen = set()
    for m in mats:
        if id(m) in seen:
            continue
        seen.add(id(m))
        uniq.append(m)
    return uniq

def _get_base_mannequin_candidates(avatar_style: Optional[str] = None) -> list[Path]:
    """
    Pick human base asset based on avatar_style.
    Expected filenames (preferred):
      - base_human_male.glb
      - base_human_female.glb
      - base_human.glb (fallback/neutral)
    Temporary compatibility fallback:
      - base_mannequin_male.glb / base_mannequin_female.glb / base_mannequin.glb
    """
    style = (avatar_style or "neutral").strip().lower()
    candidates = []
    if style in ("male", "female"):
        candidates.extend(
            [
                f"base_human_{style}.glb",
                f"base_human_{style}.gltf",
            ]
        )
    candidates.extend(["base_human.glb", "base_human.gltf"])
    if style in ("male", "female"):
        candidates.extend(
            [
                f"base_mannequin_{style}.glb",
                f"base_mannequin_{style}.gltf",
            ]
        )
    candidates.extend(["base_mannequin.glb", "base_mannequin.gltf"])
    out: list[Path] = []
    # Female style should prefer locked/base_avatar variant to preserve expected look.
    female_preferred_candidates = [
        PROJECT_ROOT_DIR / "base_avatar.glb",
        Path(__file__).resolve().parent / "base_avatar.glb",
        PROJECT_ROOT_DIR / "base_avatar_locked.glb",
        Path(__file__).resolve().parent / "base_avatar_locked.glb",
    ]
    if style == "female":
        for preferred_avatar in female_preferred_candidates:
            if preferred_avatar.exists():
                out.append(preferred_avatar)
                break
    elif USE_LOCKED_BASE_AVATAR:
        # Optional override for non-female styles in local testing.
        for preferred_avatar in female_preferred_candidates:
            if preferred_avatar.exists():
                out.append(preferred_avatar)
                break
    for name in candidates:
        p = BASE_MANNEQUIN_DIR / name
        if p.exists():
            out.append(p)
    return out


def _get_base_mannequin_path(avatar_style: Optional[str] = None) -> Optional[Path]:
    paths = _get_base_mannequin_candidates(avatar_style)
    return paths[0] if paths else None

logger = logging.getLogger(__name__)

_face_detector = "unset"
_use_opencv_fallback = False


def _try_load_mediapipe():
    global _use_opencv_fallback
    if _use_opencv_fallback:
        return False
    try:
        import mediapipe as mp
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
        from mediapipe.tasks.python.vision.core import vision_task_running_mode
        fp = get_face_model_path()
        fo = FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=fp),
            num_faces=1,
            running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        )
        global _face_detector
        _face_detector = FaceLandmarker.create_from_options(fo)
        logger.info("Face detector loaded at startup")
        return True
    except Exception as e:
        logger.warning("Face detector ikke tilgjengelig (%s), bruker OpenCV fallback", e)
        _use_opencv_fallback = True
        _face_detector = None
        return False


def _get_face_detector():
    global _face_detector
    if _face_detector == "unset":
        _try_load_mediapipe()
    return _face_detector if not _use_opencv_fallback else None


def _get_pose_detector(request_id: Optional[str] = None):
    return pose_service.get_detector(request_id=request_id)


def ensure_models_loaded() -> bool:
    """Initialize face detector at app startup."""
    ok = _try_load_mediapipe()
    logger.info("face detector init status: ok=%s", ok)
    return ok


def _face_region_opencv(rgb: np.ndarray) -> Optional[tuple]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    pad = max(20, w // 4)
    h_img, w_img = rgb.shape[:2]
    return (max(0, x - pad), max(0, y - pad), min(w_img, x + w + pad), min(h_img, y + h + pad))


def _read_image(path: str) -> Optional[np.ndarray]:
    try:
        data = Path(path).read_bytes()
    except Exception:
        return None
    # Quick format hint for debugging (does not affect decoding)
    # HEIC/HEIF commonly fails in OpenCV unless extra codecs are installed.
    if b"ftypheic" in data[:64] or b"ftypheif" in data[:64]:
        logger.warning("image appears to be HEIC/HEIF and may not decode: path=%s", path)
    img = _decode_image_bytes(data)
    if img is None:
        return None
    return _normalize_image_to_rgb_uint8(img)


def _decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is not None:
        return decoded

    # Fallback: try PIL for formats OpenCV can't decode.
    try:
        img = Image.open(io.BytesIO(data))
        # Preserve alpha if present.
        if img.mode in ("RGBA", "LA") or ("transparency" in getattr(img, "info", {})):
            img = img.convert("RGBA")
            rgba = np.array(img, dtype=np.uint8)
            # Convert RGBA -> BGRA so _normalize_image_to_rgb_uint8() yields correct RGB.
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            return bgra
        img = img.convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception:
        pass

    # Support body image payloads accidentally sent as base64 text.
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
    decoded2 = cv2.imdecode(arr2, cv2.IMREAD_UNCHANGED)
    if decoded2 is not None:
        return decoded2
    try:
        img = Image.open(io.BytesIO(raw))
        if img.mode in ("RGBA", "LA") or ("transparency" in getattr(img, "info", {})):
            img = img.convert("RGBA")
            rgba = np.array(img, dtype=np.uint8)
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        img = img.convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _normalize_image_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise AppError(
            error_code="POSE_IMAGE_DECODE_FAILED",
            message="Ustottet bildeformat for pose-inference.",
            status_code=400,
            retryable=False,
        )
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _infer_job_id_from_upload_path(path: str) -> Optional[str]:
    try:
        stem = Path(path).stem
        for marker in ("_face", "_body_front", "_body_side"):
            if marker in stem:
                return stem.split(marker, 1)[0]
        return None
    except Exception:
        return None


def estimate_skin_color_from_face(face_path: str) -> tuple[tuple[float, float, float], dict]:
    """
    Alternative 1: Estimate skin tone from the uploaded face image.
    Returns (srgb_factor_tuple, debug_info).
    Never raises; falls back to a reasonable default.
    """
    default_srgb = (0.75, 0.65, 0.55)
    debug_dir = Path(__file__).resolve().parent / "static" / "debug_inputs"
    try:
        rgb = _read_image(face_path)
        if rgb is None or rgb.size == 0:
            return default_srgb, {"method": "default_unreadable_image", "pixels_used": 0, "face_bbox": None}

        bbox = None
        try:
            region = get_face_region(rgb)
            if region:
                bbox = [int(region[0]), int(region[1]), int(region[2]), int(region[3])]
        except Exception:
            bbox = None

        # ROI: face bbox if available, otherwise central 40% crop of full image.
        if bbox:
            x1, y1, x2, y2 = bbox
            crop = rgb[y1:y2, x1:x2]
            roi_method = "face_bbox"
        else:
            h, w = rgb.shape[:2]
            x1, x2 = int(w * 0.30), int(w * 0.70)
            y1, y2 = int(h * 0.30), int(h * 0.70)
            crop = rgb[y1:y2, x1:x2]
            roi_method = "center_crop_fallback"
            bbox = [x1, y1, x2, y2]

        if crop is None or crop.size == 0:
            return default_srgb, {"method": "default_empty_roi", "pixels_used": 0, "face_bbox": bbox}

        # Sample cheek-heavy sub-regions (less hair/background than full center patch).
        ch, cw = crop.shape[:2]
        left_cheek = crop[int(ch * 0.42):int(ch * 0.72), int(cw * 0.20):int(cw * 0.42)]
        right_cheek = crop[int(ch * 0.42):int(ch * 0.72), int(cw * 0.58):int(cw * 0.80)]
        center_patch = crop[int(ch * 0.35):int(ch * 0.75), int(cw * 0.30):int(cw * 0.70)]
        patch = np.concatenate(
            [
                p.reshape(-1, 3)
                for p in (left_cheek, right_cheek, center_patch)
                if p is not None and p.size > 0
            ],
            axis=0,
        ) if any((p is not None and p.size > 0) for p in (left_cheek, right_cheek, center_patch)) else crop.reshape(-1, 3)

        # Robust skin mask:
        # - YCrCb keeps broad skin chroma span (light/dark inclusive).
        # - HSV removes extreme shadows/highlights and gray background noise.
        pixels = patch.reshape(-1, 3)
        patch_img = patch.reshape(-1, 1, 3).astype(np.uint8)
        ycrcb = cv2.cvtColor(patch_img, cv2.COLOR_RGB2YCrCb).reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(patch_img, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        Y = ycrcb[:, 0]
        Cr = ycrcb[:, 1]
        Cb = ycrcb[:, 2]
        H = hsv[:, 0]
        S = hsv[:, 1]
        V = hsv[:, 2]
        mask = (
            (Y > 20) & (Y < 245)
            & (Cr >= 112) & (Cr <= 190)
            & (Cb >= 68) & (Cb <= 148)
            & (S >= 15) & (S <= 220)
            & (V >= 25) & (V <= 245)
            & (H >= 0) & (H <= 50)
        )

        sel = pixels[mask]
        method = f"median_ycrcb_masked({roi_method})"
        if sel is None or len(sel) < 60:
            sel = pixels
            method = f"median_roi_fallback({roi_method})"

        med = np.median(sel.astype(np.float32), axis=0)
        rgb8 = [
            int(np.clip(round(float(med[0])), 18, 245)),
            int(np.clip(round(float(med[1])), 16, 235)),
            int(np.clip(round(float(med[2])), 14, 230)),
        ]
        srgb_factor = (rgb8[0] / 255.0, rgb8[1] / 255.0, rgb8[2] / 255.0)

        # Optional debug images: ROI crop + mask.
        job_id = _infer_job_id_from_upload_path(face_path)
        crop_path = None
        mask_path = None
        try:
            if job_id:
                debug_dir.mkdir(parents=True, exist_ok=True)
                crop_path = str((debug_dir / f"{job_id}_face_crop.jpg").resolve())
                mask_path = str((debug_dir / f"{job_id}_skin_mask.png").resolve())
                # Save crop as BGR for cv2.imwrite
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                # Save 1D mask for diagnostics (converted to narrow image strip).
                m = (mask.astype(np.uint8) * 255).reshape(-1, 1)
                cv2.imwrite(mask_path, m)
        except Exception:
            crop_path = None
            mask_path = None

        debug = {
            "method": method,
            "pixels_used": int(len(sel)) if sel is not None else 0,
            "face_bbox": bbox,
            "rgb8": rgb8,
            "srgb_factor": list(srgb_factor),
            "debug_face_crop_path": crop_path,
            "debug_skin_mask_path": mask_path,
        }
        return srgb_factor, debug
    except Exception:
        return default_srgb, {"method": "default_exception", "pixels_used": 0, "face_bbox": None}


def estimate_skin_color_from_body_front(
    body_rgb: Optional[np.ndarray],
    *,
    seed_rgb8: Optional[list[int]] = None,
    person_bbox_px: Optional[dict] = None,
    job_id: Optional[str] = None,
) -> tuple[Optional[tuple[float, float, float]], dict]:
    """
    Best-effort skin tone estimate from bodyFront image (neck/hands/visible skin).
    Uses a YCrCb skin mask; optionally narrows using seed skin color from face.
    Returns (srgb_factor_tuple_or_None, debug_info). Never raises.
    """
    debug_dir = Path(__file__).resolve().parent / "static" / "debug_inputs"
    try:
        if body_rgb is None or getattr(body_rgb, "size", 0) == 0:
            return None, {"method": "body_unreadable_image", "pixels_used": 0}

        rgb = body_rgb
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = 0, 0, w, h
        roi_method = "full_image"
        if isinstance(person_bbox_px, dict):
            try:
                bx1 = int(max(0, min(w, round(float(person_bbox_px.get("x1", 0.0))))))
                by1 = int(max(0, min(h, round(float(person_bbox_px.get("y1", 0.0))))))
                bx2 = int(max(0, min(w, round(float(person_bbox_px.get("x2", float(w)))))))
                by2 = int(max(0, min(h, round(float(person_bbox_px.get("y2", float(h)))))))
                if (bx2 - bx1) >= 32 and (by2 - by1) >= 32:
                    x1, y1, x2, y2 = bx1, by1, bx2, by2
                    roi_method = "person_bbox_px"
            except Exception:
                pass

        crop = rgb[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, {"method": "body_empty_roi", "pixels_used": 0, "roi": [x1, y1, x2, y2]}

        # Optional downscale for speed (keep aspect).
        ch, cw = crop.shape[:2]
        max_side = max(ch, cw)
        if max_side > 900:
            scale = 900.0 / float(max_side)
            crop = cv2.resize(crop, (int(round(cw * scale)), int(round(ch * scale))), interpolation=cv2.INTER_AREA)
            ch, cw = crop.shape[:2]

        ycrcb = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y = ycrcb[:, :, 0]
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]
        # Broadened for light AND dark skin (was Y>45, Cr 133-173, Cb 77-127).
        mask = (Y > 25) & (Y < 240) & (Cr >= 115) & (Cr <= 185) & (Cb >= 70) & (Cb <= 145)
        method = f"median_ycrcb_masked({roi_method})"

        # If we have a face-derived seed, bias selection toward similar chroma to preserve undertones.
        if seed_rgb8 and len(seed_rgb8) == 3:
            try:
                seed = np.array([[seed_rgb8]], dtype=np.uint8)
                seed_ycrcb = cv2.cvtColor(seed, cv2.COLOR_RGB2YCrCb).astype(np.float32)[0, 0]
                sCr = float(seed_ycrcb[1])
                sCb = float(seed_ycrcb[2])
                tight = mask & (np.abs(Cr - sCr) <= 25.0) & (np.abs(Cb - sCb) <= 25.0)
                if int(tight.sum()) >= 150:
                    mask = tight
                    method = f"median_ycrcb_seeded_tight({roi_method})"
                else:
                    loose = mask & (np.abs(Cr - sCr) <= 45.0) & (np.abs(Cb - sCb) <= 45.0)
                    if int(loose.sum()) >= 150:
                        mask = loose
                        method = f"median_ycrcb_seeded_loose({roi_method})"
            except Exception:
                pass

        m8 = (mask.astype(np.uint8) * 255)
        try:
            kernel = np.ones((3, 3), np.uint8)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel, iterations=1)
            m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            pass

        sel = crop[m8 > 0].reshape(-1, 3)
        if sel is None or len(sel) < 250:
            return None, {"method": "body_insufficient_skin_pixels", "pixels_used": int(len(sel)) if sel is not None else 0, "roi": [x1, y1, x2, y2]}

        med = np.median(sel.astype(np.float32), axis=0)
        rgb8 = [
            int(np.clip(round(float(med[0])), 0, 255)),
            int(np.clip(round(float(med[1])), 0, 255)),
            int(np.clip(round(float(med[2])), 0, 255)),
        ]
        srgb_factor = (rgb8[0] / 255.0, rgb8[1] / 255.0, rgb8[2] / 255.0)

        mask_path = None
        crop_path = None
        try:
            if job_id:
                debug_dir.mkdir(parents=True, exist_ok=True)
                crop_path = str((debug_dir / f"{job_id}_body_crop.jpg").resolve())
                mask_path = str((debug_dir / f"{job_id}_body_skin_mask.png").resolve())
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                cv2.imwrite(mask_path, m8)
        except Exception:
            crop_path = None
            mask_path = None

        return srgb_factor, {
            "method": method,
            "pixels_used": int(len(sel)),
            "roi_method": roi_method,
            "roi": [int(x1), int(y1), int(x2), int(y2)],
            "rgb8": rgb8,
            "srgb_factor": list(srgb_factor),
            "debug_body_crop_path": crop_path,
            "debug_body_skin_mask_path": mask_path,
        }
    except Exception:
        return None, {"method": "body_exception", "pixels_used": 0}


# Skin material heuristics (C): name or mesh contains these.
SKIN_MATERIAL_KEYWORDS = ("skin", "skin_m", "body", "head", "face", "hands", "arm", "leg", "torso", "mannequin", "character")
SKIN_INCLUDE_MATERIAL_KWS = ("skin", "body")
SKIN_EXCLUDE_MATERIAL_KWS = ("hair", "brow", "beard", "eye", "iris", "cornea", "teeth", "tongue", "cloth", "underwear")


def _material_texture_uri(mat) -> Optional[str]:
    try:
        bct = getattr(mat, "baseColorTexture", None)
        if bct is not None:
            uri = getattr(bct, "uri", None)
            if uri:
                return str(uri)
    except Exception:
        pass
    try:
        img = getattr(mat, "image", None)
        if img is None:
            return None
        fn = getattr(img, "filename", None)
        if fn:
            return str(fn)
        return "embedded://image"
    except Exception:
        return None


def _collect_material_entries(scene: trimesh.Scene) -> list[dict]:
    """Collect material entries with mesh/node usage for debug and selection."""
    entries: dict[int, dict] = {}
    for geom_name, g in scene.geometry.items():
        mat = _geometry_material(g)
        if mat is None:
            continue
        key = id(mat)
        if key not in entries:
            entries[key] = {
                "id": key,
                "mat": mat,
                "materialName": _material_name(mat),
                "meshNames": [],
                "nodeNames": [],
                "weight": 0.0,
            }
        entries[key]["meshNames"].append(str(geom_name))
        try:
            entries[key]["weight"] += float(len(getattr(g, "faces", []))) if hasattr(g, "faces") else float(len(getattr(g, "vertices", [])))
        except Exception:
            entries[key]["weight"] += 1.0

    try:
        for node in scene.graph.nodes:
            try:
                _, geom_name = scene.graph.get(node)
            except Exception:
                continue
            if geom_name not in scene.geometry:
                continue
            mat = _geometry_material(scene.geometry[geom_name])
            if mat is None:
                continue
            key = id(mat)
            if key in entries:
                entries[key]["nodeNames"].append(str(node))
    except Exception:
        pass

    out = []
    for it in entries.values():
        it["meshNames"] = sorted(set(it["meshNames"]))
        it["nodeNames"] = sorted(set(it["nodeNames"]))
        out.append(it)
    return out


def _detect_skin_material_ids(entries: list[dict]) -> tuple[set[int], bool]:
    """
    Detect actual skin materials using material/mesh/node names.
    If no match, choose dominant body materials (major surface) as pragmatic fallback.
    """
    keywords = ("skin", "body", "head", "face", "arm", "hand", "leg", "torso", "neck", "hip", "thigh", "calf")
    selected: set[int] = set()
    found_by_skin_heuristics = False
    total_weight = max(1.0, sum(float(e.get("weight", 0.0)) for e in entries))

    for e in entries:
        mn = str(e.get("materialName", "")).lower()
        mesh_names = [str(x).lower() for x in e.get("meshNames", [])]
        node_names = [str(x).lower() for x in e.get("nodeNames", [])]
        name_hit = any(k in mn for k in keywords)
        mesh_hit = any(any(k in n for k in keywords) for n in mesh_names)
        node_hit = any(any(k in n for k in keywords) for n in node_names)
        if name_hit or mesh_hit or node_hit:
            found_by_skin_heuristics = True
            selected.add(int(e["id"]))

    if not selected:
        # Pragmatic fallback: dominant body-like materials (no hardcoded Azul/Blanco/Negro).
        for e in entries:
            ratio = float(e.get("weight", 0.0)) / total_weight
            if ratio >= 0.20:
                selected.add(int(e["id"]))
    return selected, found_by_skin_heuristics


def _material_has_normal_map(mat) -> bool:
    try:
        return bool(getattr(mat, "normalTexture", None) or getattr(mat, "normalMap", None))
    except Exception:
        return False


def _material_list_for_debug(scene: trimesh.Scene) -> list[dict]:
    out = []
    for e in _collect_material_entries(scene):
        mat = e["mat"]
        out.append(
            {
                "materialName": str(e.get("materialName")),
                "meshNames": list(e.get("meshNames", [])),
                "baseColorFactor": sanitize_for_json(_material_base_color_factor(mat)),
                "metallicFactor": sanitize_for_json(getattr(mat, "metallicFactor", getattr(mat, "metalness", None))),
                "roughnessFactor": sanitize_for_json(getattr(mat, "roughnessFactor", getattr(mat, "roughness", None))),
                "hasBaseColorTexture": bool(getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)),
                "baseColorTextureUri": _material_texture_uri(mat),
                "hasNormalMap": bool(_material_has_normal_map(mat)),
            }
        )
    return out


def _apply_fixed_human_materials(scene: trimesh.Scene, skin_srgb: tuple[float, float, float]) -> tuple[list[str], list[dict], bool, bool]:
    """
    Human visual mode:
    - Do NOT sample skin from image
    - Apply fixed natural skin tone to detected skin materials
    - Keep baseColor textures for realism (no forced removal here)
    """
    r_s, g_s, b_s = float(skin_srgb[0]), float(skin_srgb[1]), float(skin_srgb[2])
    factor = [
        _srgb_to_linear_channel(r_s),
        _srgb_to_linear_channel(g_s),
        _srgb_to_linear_channel(b_s),
        1.0,
    ]
    changed: list[str] = []
    rows: list[dict] = []
    try:
        entries = _collect_material_entries(scene)
        # Hvis scenen mangler teksturer helt, ikke rør skin-materialer (la base-material stå).
        has_any_texture = any(
            bool(_material_texture_uri(e.get("mat"))) or bool(getattr(e.get("mat"), "image", None))
            for e in entries
        )
        if not has_any_texture:
            logger.warning("apply_skin_color_to_scene: no textures detected on any material; skipping skin color application")
            return [], [], True, False, False
        skin_ids, found_by_skin_heuristics = _detect_skin_material_ids(entries)
        skin_not_found = not found_by_skin_heuristics
        changed_any = False
        visible_any = False

        for e in entries:
            mat = e["mat"]
            mat_id = int(e["id"])
            name = str(e.get("materialName", ""))
            n = name.lower()
            before = _material_base_color_factor(mat)
            has_tex_before = bool(getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None))
            applied = False
            is_skin_candidate = bool(mat_id in skin_ids)

            if is_skin_candidate:
                try:
                    if hasattr(mat, "baseColorFactor"):
                        mat.baseColorFactor = [float(factor[0]), float(factor[1]), float(factor[2]), 1.0]
                    if hasattr(mat, "metallicFactor"):
                        mat.metallicFactor = 0.0
                    if hasattr(mat, "metalness"):
                        mat.metalness = 0.0
                    if hasattr(mat, "roughnessFactor"):
                        mat.roughnessFactor = 0.72
                    if hasattr(mat, "roughness"):
                        mat.roughness = 0.72
                    applied = True
                    changed_any = True
                    visible_any = visible_any or (not has_tex_before)
                except Exception:
                    pass

            # Facial realism tweaks (eyes/lips) if materials exist.
            try:
                if any(k in n for k in ("eye", "iris", "pupil", "sclera")):
                    if hasattr(mat, "metallicFactor"):
                        mat.metallicFactor = 0.0
                    if hasattr(mat, "roughnessFactor"):
                        mat.roughnessFactor = 0.18
                    if hasattr(mat, "roughness"):
                        mat.roughness = 0.18
                if any(k in n for k in ("lip", "mouth")):
                    if hasattr(mat, "baseColorFactor"):
                        lip = [factor[0] * 0.86, factor[1] * 0.72, factor[2] * 0.72, 1.0]
                        mat.baseColorFactor = [float(lip[0]), float(lip[1]), float(lip[2]), 1.0]
                    if hasattr(mat, "roughnessFactor"):
                        mat.roughnessFactor = 0.45
            except Exception:
                pass

            after = _material_base_color_factor(mat)
            if applied:
                changed.append(name)
            rows.append(
                {
                    "materialName": name,
                    "meshNames": list(e.get("meshNames", [])),
                    "baseColorFactorBefore": sanitize_for_json(before),
                    "baseColorFactorAfter": sanitize_for_json(after),
                    "hasBaseColorTexture": has_tex_before,
                    "baseColorTextureUri": _material_texture_uri(mat),
                    "skinCandidate": is_skin_candidate,
                    "applied": bool(applied),
                }
            )

        return sorted(set(changed)), rows, skin_not_found, bool(changed_any and (visible_any or True))
    except Exception:
        return [], [], True, False


def _material_has_normal_map(mat) -> bool:
    try:
        return bool(getattr(mat, "normalTexture", None) or getattr(mat, "normalMap", None))
    except Exception:
        return False


def _material_list_for_debug(scene: trimesh.Scene) -> list[dict]:
    out = []
    for e in _collect_material_entries(scene):
        mat = e["mat"]
        out.append(
            {
                "materialName": str(e.get("materialName")),
                "meshNames": list(e.get("meshNames", [])),
                "baseColorFactor": sanitize_for_json(_material_base_color_factor(mat)),
                "metallicFactor": sanitize_for_json(getattr(mat, "metallicFactor", getattr(mat, "metalness", None))),
                "roughnessFactor": sanitize_for_json(getattr(mat, "roughnessFactor", getattr(mat, "roughness", None))),
                "hasBaseColorTexture": bool(getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)),
                "baseColorTextureUri": _material_texture_uri(mat),
                "hasNormalMap": bool(_material_has_normal_map(mat)),
            }
        )
    return out


def _apply_fixed_human_materials(scene: trimesh.Scene, skin_srgb: tuple[float, float, float]) -> tuple[list[str], list[dict], bool, bool]:
    """
    Human visual mode:
    - Do NOT sample skin from image
    - Apply fixed natural skin tone to detected skin materials
    - Keep baseColor textures for realism (no forced removal here)
    """
    r_s, g_s, b_s = float(skin_srgb[0]), float(skin_srgb[1]), float(skin_srgb[2])
    factor = [
        _srgb_to_linear_channel(r_s),
        _srgb_to_linear_channel(g_s),
        _srgb_to_linear_channel(b_s),
        1.0,
    ]
    changed: list[str] = []
    rows: list[dict] = []
    try:
        entries = _collect_material_entries(scene)
        skin_ids, found_by_skin_heuristics = _detect_skin_material_ids(entries)
        skin_not_found = not found_by_skin_heuristics
        changed_any = False
        visible_any = False

        for e in entries:
            mat = e["mat"]
            mat_id = int(e["id"])
            name = str(e.get("materialName", ""))
            n = name.lower()
            before = _material_base_color_factor(mat)
            has_tex_before = bool(getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None))
            applied = False
            is_skin_candidate = bool(mat_id in skin_ids)

            if is_skin_candidate:
                try:
                    if hasattr(mat, "baseColorFactor"):
                        mat.baseColorFactor = [float(factor[0]), float(factor[1]), float(factor[2]), 1.0]
                    if hasattr(mat, "metallicFactor"):
                        mat.metallicFactor = 0.0
                    if hasattr(mat, "metalness"):
                        mat.metalness = 0.0
                    if hasattr(mat, "roughnessFactor"):
                        mat.roughnessFactor = 0.72
                    if hasattr(mat, "roughness"):
                        mat.roughness = 0.72
                    applied = True
                    changed_any = True
                    # visible if no baseColor texture OR factor actually changed
                    visible_any = visible_any or (not has_tex_before)
                except Exception:
                    pass

            # Facial realism tweaks (eyes/lips) if materials exist.
            try:
                if any(k in n for k in ("eye", "iris", "pupil", "sclera")):
                    if hasattr(mat, "metallicFactor"):
                        mat.metallicFactor = 0.0
                    if hasattr(mat, "roughnessFactor"):
                        mat.roughnessFactor = 0.18
                    if hasattr(mat, "roughness"):
                        mat.roughness = 0.18
                if any(k in n for k in ("lip", "mouth")):
                    if hasattr(mat, "baseColorFactor"):
                        lip = [factor[0] * 0.86, factor[1] * 0.72, factor[2] * 0.72, 1.0]
                        mat.baseColorFactor = [float(lip[0]), float(lip[1]), float(lip[2]), 1.0]
                    if hasattr(mat, "roughnessFactor"):
                        mat.roughnessFactor = 0.45
            except Exception:
                pass

            after = _material_base_color_factor(mat)
            if applied:
                changed.append(name)
            rows.append(
                {
                    "materialName": name,
                    "meshNames": list(e.get("meshNames", [])),
                    "baseColorFactorBefore": sanitize_for_json(before),
                    "baseColorFactorAfter": sanitize_for_json(after),
                    "hasBaseColorTexture": has_tex_before,
                    "baseColorTextureUri": _material_texture_uri(mat),
                    "skinCandidate": is_skin_candidate,
                    "applied": bool(applied),
                }
            )

        return sorted(set(changed)), rows, skin_not_found, bool(changed_any and (visible_any or True))
    except Exception:
        return [], [], True, False


def apply_skin_color_to_scene(scene: trimesh.Scene, rgb_srgb: tuple[float, float, float]) -> tuple[list[str], list[dict], bool, bool, bool]:
    """
    Apply skin tone to detected skin materials only.
    Option A (deterministic): remove baseColorTexture for skin materials.
    Returns (changed_names, material_debug_rows, skinMaterialNotFound, skinAppliedVisible, skinTextureRemoved).
    """
    r_s, g_s, b_s = float(rgb_srgb[0]), float(rgb_srgb[1]), float(rgb_srgb[2])
    # Requirement (3): convert sRGB -> linear for PBR baseColorFactor.
    factor = [
        _srgb_to_linear_channel(r_s),
        _srgb_to_linear_channel(g_s),
        _srgb_to_linear_channel(b_s),
        1.0,
    ]
    changed: list[str] = []
    rows: list[dict] = []
    try:
        entries = _collect_material_entries(scene)
        selected_ids, found_by_skin_heuristics = _detect_skin_material_ids(entries)
        skin_not_found = not found_by_skin_heuristics
        texture_removed_any = False
        changed_any = False

        for e in entries:
            mat = e["mat"]
            mat_name = str(e["materialName"])
            mat_name_low = mat_name.lower()
            mesh_names_low = [str(x).lower() for x in e.get("meshNames", [])]
            node_names_low = [str(x).lower() for x in e.get("nodeNames", [])]
            include_hit = any(k in mat_name_low for k in SKIN_INCLUDE_MATERIAL_KWS)
            mesh_hit = any(any(k in n for k in ("skin", "body", "head", "face", "neck", "arm", "leg", "torso")) for n in mesh_names_low)
            node_hit = any(any(k in n for k in ("skin", "body", "head", "face", "neck", "arm", "leg", "torso")) for n in node_names_low)
            exclude_hit = any(k in mat_name_low for k in SKIN_EXCLUDE_MATERIAL_KWS) or any(
                any(k in n for k in SKIN_EXCLUDE_MATERIAL_KWS) for n in (mesh_names_low + node_names_low)
            )
            skin_candidate = bool((int(e["id"]) in selected_ids or include_hit or mesh_hit or node_hit) and not exclude_hit)
            before = _material_base_color_factor(mat)
            has_tex_before = bool(getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None))
            tex_uri = _material_texture_uri(mat)
            after = before
            has_tex_after = has_tex_before
            applied = False

            if skin_candidate:
                skin_not_found = False
                try:
                    if hasattr(mat, "baseColorFactor"):
                        mat.baseColorFactor = [float(factor[0]), float(factor[1]), float(factor[2]), 1.0]
                    # Force a tiny embedded texture so baseColorFactor is guaranteed visible in viewers.
                    try:
                        mat.image = _get_white_1x1_texture().copy()
                    except Exception:
                        pass
                    # Keep base textures; only tint skin and preserve details.
                    if hasattr(mat, "metallicFactor"):
                        mat.metallicFactor = 0.0
                    if hasattr(mat, "metalness"):
                        mat.metalness = 0.0
                    if hasattr(mat, "roughnessFactor"):
                        mat.roughnessFactor = 0.80
                    if hasattr(mat, "roughness"):
                        mat.roughness = 0.80
                    if hasattr(mat, "emissiveFactor"):
                        mat.emissiveFactor = [0.0, 0.0, 0.0]
                    if hasattr(mat, "emissive"):
                        try:
                            mat.emissive = [0.0, 0.0, 0.0]
                        except Exception:
                            pass
                    after = _material_base_color_factor(mat)
                    has_tex_after = bool(getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None))
                    texture_removed_any = texture_removed_any or (has_tex_before and not has_tex_after)
                    applied = True
                    changed_any = changed_any or (sanitize_for_json(before) != sanitize_for_json(after))
                    changed.append(mat_name)
                    logger.info(
                        "skin material updated: name=%s mesh=%s nodes=%s hasTexBefore=%s hasTexAfter=%s before_baseColor=%s after_baseColor=%s",
                        e["materialName"],
                        e.get("meshNames"),
                        e.get("nodeNames"),
                        has_tex_before,
                        has_tex_after,
                        before,
                        after,
                    )
                except Exception:
                    pass

            rows.append(
                {
                    "materialName": str(e["materialName"]),
                    "meshNames": list(e.get("meshNames", [])),
                    "nodeNames": list(e.get("nodeNames", [])),
                    "baseColorFactorBefore": sanitize_for_json(before),
                    "baseColorFactorAfter": sanitize_for_json(after),
                    "hasBaseColorTexture": bool(has_tex_before),
                    "hasBaseColorTextureAfter": bool(has_tex_after),
                    "baseColorTextureUri": tex_uri,
                    "baseColorFactorSet": bool(applied),
                    "skinCandidate": bool(skin_candidate),
                    "skippedByRule": bool((not skin_candidate) and exclude_hit),
                    "applied": bool(applied),
                }
            )

        skin_applied_visible = bool(changed_any)
        return sorted(set(changed)), rows, skin_not_found, skin_applied_visible, bool(texture_removed_any)
    except Exception:
        return [], [], True, False, False


def enforce_male_hair_black(scene: trimesh.Scene, avatar_style: Optional[str]) -> list[str]:
    """
    Keep male hair dark/black regardless of skin tint operations.
    """
    if str(avatar_style or "").strip().lower() != "male":
        return []
    forced: list[str] = []
    try:
        entries = _collect_material_entries(scene)
        for e in entries:
            mat = e.get("mat")
            if mat is None:
                continue
            mat_name = str(e.get("materialName", "")).lower()
            mesh_names = " ".join(str(x).lower() for x in e.get("meshNames", []))
            node_names = " ".join(str(x).lower() for x in e.get("nodeNames", []))
            hay = f"{mat_name} {mesh_names} {node_names}"
            is_hair_like = any(k in hay for k in ("hair", "scalp", "eyebrow", "brow", "beard", "mustache", "moustache"))
            if not is_hair_like:
                continue
            try:
                dark = [
                    _srgb_to_linear_channel(0.03),
                    _srgb_to_linear_channel(0.03),
                    _srgb_to_linear_channel(0.03),
                    1.0,
                ]
                if hasattr(mat, "baseColorFactor"):
                    mat.baseColorFactor = [float(dark[0]), float(dark[1]), float(dark[2]), 1.0]
                # Force hair albedo to black so light/white textures cannot override.
                try:
                    mat.image = _get_black_1x1_texture().copy()
                except Exception:
                    pass
                try:
                    if hasattr(mat, "baseColorTexture"):
                        mat.baseColorTexture = _get_black_1x1_texture().copy()
                except Exception:
                    pass
                if hasattr(mat, "metallicFactor"):
                    mat.metallicFactor = 0.0
                if hasattr(mat, "metalness"):
                    mat.metalness = 0.0
                if hasattr(mat, "roughnessFactor"):
                    mat.roughnessFactor = 0.78
                if hasattr(mat, "roughness"):
                    mat.roughness = 0.78
                forced.append(str(e.get("materialName", "")))
            except Exception:
                continue
    except Exception:
        return []
    return sorted(set(forced))


def _material_debug_summary(scene: trimesh.Scene, top_n: int = 8) -> list[dict]:
    """Return a small JSON-safe summary of most-used materials."""
    try:
        weights = _material_usage_weights(scene)
        ranked = sorted(weights.values(), key=lambda x: float(x.get("weight", 0.0)), reverse=True)[: max(1, int(top_n))]
        out = []
        for it in ranked:
            mat = it["mat"]
            out.append(
                {
                    "name": _material_name(mat),
                    "weight": float(it.get("weight", 0.0)),
                    "baseColor": sanitize_for_json(_material_base_color_factor(mat)),
                    "metallic": sanitize_for_json(getattr(mat, "metallicFactor", getattr(mat, "metalness", None))),
                    "roughness": sanitize_for_json(getattr(mat, "roughnessFactor", getattr(mat, "roughness", None))),
                    "alphaMode": sanitize_for_json(getattr(mat, "alphaMode", None)),
                }
            )
        return out
    except Exception:
        return []


def apply_skin_color_to_glb(glb_path: str, color_factor_srgb: list[float], request_id: Optional[str] = None) -> dict:
    """
    Post-process an exported GLB and apply PBR baseColorFactor to a likely "skin" material.
    Uses pygltflib FIRST (writes correct float [0,1] format). Trimesh stores uint8 and exports wrong.
    Best-effort: never raises (caller should wrap anyway).
    """
    keywords = SKIN_MATERIAL_KEYWORDS
    r_s, g_s, b_s = float(color_factor_srgb[0]), float(color_factor_srgb[1]), float(color_factor_srgb[2])
    # Requirement (3): sRGB -> linear for glTF PBR.
    factor = [
        _srgb_to_linear_channel(r_s),
        _srgb_to_linear_channel(g_s),
        _srgb_to_linear_channel(b_s),
        1.0,
    ]
    updated = []

    # PRIMARY: pygltflib writes baseColorFactor as float [0,1] (glTF spec). Trimesh stores uint8 and exports wrong.
    try:
        from pygltflib import GLTF2, PbrMetallicRoughness

        gltf = GLTF2().load(glb_path)
        if gltf.materials:
            indices = []
            for i, m in enumerate(gltf.materials):
                n = (m.name or "").lower()
                if any(k in n for k in keywords):
                    indices.append(i)
            if not indices:
                indices = list(range(min(3, len(gltf.materials))))

            for idx in indices:
                mat = gltf.materials[idx]
                if mat.pbrMetallicRoughness is None:
                    mat.pbrMetallicRoughness = PbrMetallicRoughness()
                mat.pbrMetallicRoughness.baseColorFactor = [float(factor[0]), float(factor[1]), float(factor[2]), 1.0]
                try:
                    if hasattr(mat.pbrMetallicRoughness, "baseColorTexture"):
                        delattr(mat.pbrMetallicRoughness, "baseColorTexture")
                except Exception:
                    try:
                        mat.pbrMetallicRoughness.baseColorTexture = None
                    except Exception:
                        pass
                updated.append(mat.name or f"material[{idx}]")
            gltf.save(glb_path)
            logger.info("skin postprocess applied (pygltflib): request_id=%s glb=%s factor=%s material=%s", request_id, glb_path, factor, updated)
            return {"applied": True, "method": "pygltflib", "materials": updated}
    except Exception as e:
        logger.warning("skin postprocess pygltflib failed request_id=%s err=%s", request_id, e)

    # Fallback: trimesh (may export baseColorFactor as uint8, causing grey in viewer).
    try:
        loaded = trimesh.load(glb_path, force="scene")
        scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)

        # Collect unique materials in stable order.
        mats = []
        seen = set()
        for _, mat in _iter_unique_scene_materials(scene):
            if id(mat) in seen:
                continue
            seen.add(id(mat))
            mats.append(mat)

        targets = [m for m in mats if any(k in _material_name(m).lower() for k in keywords)]
        if not targets and mats:
            targets = mats[:3]

        if not targets:
            logger.warning("skin postprocess: request_id=%s no materials found in glb=%s", request_id, glb_path)
            return {"applied": False, "method": "trimesh", "materials": []}

        for target in targets:
            if hasattr(target, "baseColorFactor"):
                target.baseColorFactor = [float(factor[0]), float(factor[1]), float(factor[2]), 1.0]
            if hasattr(target, "baseColorTexture"):
                try:
                    delattr(target, "baseColorTexture")
                except Exception:
                    target.baseColorTexture = None
            if hasattr(target, "image"):
                target.image = None
            if hasattr(target, "alphaMode"):
                target.alphaMode = "OPAQUE"
            if hasattr(target, "transparent"):
                target.transparent = False
            if hasattr(target, "opacity"):
                target.opacity = 1.0
            if hasattr(target, "diffuse"):
                target.diffuse = [int(round(r_s * 255)), int(round(g_s * 255)), int(round(b_s * 255)), 255]
            if hasattr(target, "ambient"):
                target.ambient = [int(round(r_s * 255)), int(round(g_s * 255)), int(round(b_s * 255)), 255]
            updated.append(_material_name(target))
        scene.export(glb_path, file_type="glb")
        logger.info("skin postprocess applied (trimesh): request_id=%s glb=%s factor=%s material=%s", request_id, glb_path, factor, updated)
        return {"applied": True, "method": "trimesh", "materials": updated}
    except Exception as e:
        logger.warning("skin postprocess trimesh failed request_id=%s err=%s", request_id, e)
        return {"applied": False, "method": "failed", "materials": []}


def _numpy_to_mp_image(rgb: np.ndarray):
    import mediapipe as mp
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _scene_bbox(scene: trimesh.Scene) -> dict:
    geoms = [g for g in scene.geometry.values() if hasattr(g, "bounds")]

    if not geoms:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    # Each g.bounds is shape (2, 3). Stack into (2N, 3) and reduce over axis 0.
    all_bounds = np.vstack([np.asarray(g.bounds, dtype=float).reshape(2, 3) for g in geoms])
    bbox_min = all_bounds.min(axis=0)
    bbox_max = all_bounds.max(axis=0)
    size = np.asarray(bbox_max - bbox_min, dtype=float).reshape(-1)

    # Hvis noe gikk galt og vi ikke fikk 3 dimensjoner
    if size.size < 3:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    return {
        "x": float(size[0]),
        "y": float(size[1]),
        "z": float(size[2])
    }

def _landmark_px(landmarks, idx: int, w: int, h: int):
    try:
        lm = landmarks[idx]
        return np.array([float(lm.x) * w, float(lm.y) * h], dtype=np.float32)
    except Exception:
        return None


def _estimate_body_fit_from_pose(body_rgb: np.ndarray, request_id: Optional[str] = None):
    """Estimate body scales + debug measurements from pose keypoints (front view)."""
    if body_rgb is None:
        logger.error("POSE FAILED – no keypoints detected (image is None)")
        raise RuntimeError("POSE_FAILED: no keypoints detected (image is None)")

    h, w = body_rgb.shape[:2]
    ch = body_rgb.shape[2] if body_rgb.ndim == 3 else 1
    mode = "RGBA" if ch == 4 else "RGB" if ch == 3 else f"OTHER({ch})"
    logger.info("pose input image shape: height=%s width=%s channels=%s", h, w, ch)
    logger.info("pose input image mode: %s", mode)

    if USE_HARDCODED_BODY_MEASUREMENTS:
        shoulder_w = HARDCODED_BODY_MEASUREMENTS["shoulderWidthPx"]
        hip_w = HARDCODED_BODY_MEASUREMENTS["hipWidthPx"]
        torso_h = HARDCODED_BODY_MEASUREMENTS["torsoLengthPx"]
        leg_h = HARDCODED_BODY_MEASUREMENTS["legLengthPx"]
        body_h = HARDCODED_BODY_MEASUREMENTS["totalHeightPx"]

        shoulder_ref_px = REF_SHOULDER_RATIO * body_h
        hip_ref_px = REF_HIP_RATIO * body_h
        torso_ref_px = REF_TORSO_RATIO * body_h
        leg_ref_px = REF_LEG_RATIO * body_h
        total_height_ref_px = REF_TOTAL_HEIGHT_RATIO * float(h)

        shoulder_scale_raw = shoulder_w / max(shoulder_ref_px, 1.0)
        hip_scale_raw = hip_w / max(hip_ref_px, 1.0)
        torso_scale_raw = torso_h / max(torso_ref_px, 1.0)
        leg_scale_raw = leg_h / max(leg_ref_px, 1.0)
        height_scale_raw = body_h / max(total_height_ref_px, 1.0)

        shoulder_scale = _clamp(shoulder_scale_raw, 0.82, 1.22)
        hip_scale = _clamp(hip_scale_raw, 0.85, 1.20)
        leg_scale = _clamp(leg_scale_raw, 0.90, 1.15)
        torso_scale = _clamp(torso_scale_raw, 0.90, 1.15)
        waist_scale = _clamp(0.6 * hip_scale + 0.4 * shoulder_scale, 0.86, 1.16)
        chest_scale = _clamp(0.9 * shoulder_scale + 0.1 * waist_scale, 0.86, 1.18)
        height_scale = _clamp(height_scale_raw, 0.93, 1.10)

        scales = {
            "heightScale": float(height_scale),
            "shoulderWidthScale": float(shoulder_scale),
            "chestScale": float(chest_scale),
            "waistScale": float(waist_scale),
            "hipScale": float(hip_scale),
            "legLengthScale": float(leg_scale),
            "torsoLengthScale": float(torso_scale),
        }
        measurements = {
            "shoulderWidthPx": round(float(shoulder_w), 2),
            "hipWidthPx": round(float(hip_w), 2),
            "torsoLengthPx": round(float(torso_h), 2),
            "legLengthPx": round(float(leg_h), 2),
            "totalHeightPx": round(float(body_h), 2),
            "totalHeightRatio": round(float(body_h / max(float(h), 1.0)), 4),
        }
        references = {
            "shoulderWidthPxRef": round(float(shoulder_ref_px), 2),
            "hipWidthPxRef": round(float(hip_ref_px), 2),
            "torsoLengthPxRef": round(float(torso_ref_px), 2),
            "legLengthPxRef": round(float(leg_ref_px), 2),
            "totalHeightPxRef": round(float(total_height_ref_px), 2),
        }
        formulas = {
            "shoulderWidthScale": "shoulderWidthScale = shoulderWidthPx / shoulderWidthPxRef",
            "hipScale": "hipScale = hipWidthPx / hipWidthPxRef",
            "torsoLengthScale": "torsoLengthScale = torsoLengthPx / torsoLengthPxRef",
            "legLengthScale": "legLengthScale = legLengthPx / legLengthPxRef",
            "heightScale": "heightScale = totalHeightPx / totalHeightPxRef",
        }
        logger.info("POSE BYPASS ACTIVE: using hardcoded body measurements %s", HARDCODED_BODY_MEASUREMENTS)
        logger.info("pose raw measurements(px): %s", measurements)
        logger.info("pose reference measurements(px): %s", references)
        logger.info("pose scale formulas: %s", formulas)
        logger.info(
            "pose scale raw values: shoulder=%.4f hip=%.4f torso=%.4f leg=%.4f height=%.4f",
            shoulder_scale_raw,
            hip_scale_raw,
            torso_scale_raw,
            leg_scale_raw,
            height_scale_raw,
        )
        logger.info("pose scale clamped values: %s", scales)
        debug_pose = {
            "keypoints_count": 33,
            "pose_confidence": 1.0,
            "person_bbox_px": {"x1": 0.0, "y1": 0.0, "x2": float(w), "y2": float(h)},
        }
        return scales, measurements, references, formulas, debug_pose

    detector = _get_pose_detector(request_id=request_id)
    try:
        logger.info("pose model backend used: %s", pose_service.model_name)
        result = detector.process(body_rgb)
    except Exception as e:
        logger.exception("POSE_INFERENCE_FAILED during detector.process request_id=%s", request_id)
        raise AppError(
            error_code="POSE_INFERENCE_FAILED",
            message="Kroppsanalyse feilet under pose-inference.",
            status_code=503,
            details={"request_id": request_id, "exception": type(e).__name__},
            retryable=True,
        ) from e
    try:
        # Support both backends:
        # - mediapipe.solutions.pose: result.pose_landmarks.landmark (single pose)
        # - mediapipe.tasks PoseLandmarker: result.pose_landmarks -> list[list[landmark]]
        landmarks = None
        if hasattr(result, "pose_landmarks") and result.pose_landmarks:
            pl = result.pose_landmarks
            if isinstance(pl, list):
                landmarks = pl[0] if len(pl) > 0 else None
            else:
                landmarks = getattr(pl, "landmark", None)
        if not landmarks:
            logger.error("POSE FAILED – no keypoints detected")
            raise RuntimeError("POSE_NO_KEYPOINTS: no keypoints detected")

        lms = landmarks
        logger.info("pose keypoints count: %s", len(lms))
        vis_vals = [float(getattr(lm, "visibility", 0.0)) for lm in lms]
        confidence = float(sum(vis_vals) / len(vis_vals)) if vis_vals else 0.0
        logger.info("pose detection confidence score: %.4f", confidence)
        if confidence < POSE_MIN_CONFIDENCE:
            logger.error("POSE FAILED – low confidence (%.4f < %.2f)", confidence, POSE_MIN_CONFIDENCE)
            raise RuntimeError("POSE_NO_KEYPOINTS: low confidence")
        xs = [float(lm.x) * w for lm in lms]
        ys = [float(lm.y) * h for lm in lms]
        person_bbox = {
            "x1": round(max(0.0, min(xs)), 2),
            "y1": round(max(0.0, min(ys)), 2),
            "x2": round(min(float(w), max(xs)), 2),
            "y2": round(min(float(h), max(ys)), 2),
        }
        logger.info("detected person bbox(px): %s", person_bbox)
        logger.info(
            "all pose keypoints px: %s",
            [[i, round(float(lm.x) * w, 2), round(float(lm.y) * h, 2), round(float(getattr(lm, 'visibility', 0.0)), 3)] for i, lm in enumerate(lms)],
        )

        l_sh = _landmark_px(lms, 11, w, h)
        r_sh = _landmark_px(lms, 12, w, h)
        l_hip = _landmark_px(lms, 23, w, h)
        r_hip = _landmark_px(lms, 24, w, h)
        l_knee = _landmark_px(lms, 25, w, h)
        r_knee = _landmark_px(lms, 26, w, h)
        l_ank = _landmark_px(lms, 27, w, h)
        r_ank = _landmark_px(lms, 28, w, h)
        l_wrist = _landmark_px(lms, 15, w, h)
        r_wrist = _landmark_px(lms, 16, w, h)
        nose = _landmark_px(lms, 0, w, h)
        required = {
            "left_shoulder": l_sh,
            "right_shoulder": r_sh,
            "left_hip": l_hip,
            "right_hip": r_hip,
            "left_knee": l_knee,
            "right_knee": r_knee,
            "left_ankle": l_ank,
            "right_ankle": r_ank,
            "head": nose,
        }

        logger.info(
            "pose keypoints px: left_shoulder=%s right_shoulder=%s left_hip=%s right_hip=%s left_knee=%s left_ankle=%s",
            None if l_sh is None else [round(float(l_sh[0]), 2), round(float(l_sh[1]), 2)],
            None if r_sh is None else [round(float(r_sh[0]), 2), round(float(r_sh[1]), 2)],
            None if l_hip is None else [round(float(l_hip[0]), 2), round(float(l_hip[1]), 2)],
            None if r_hip is None else [round(float(r_hip[0]), 2), round(float(r_hip[1]), 2)],
            None if l_knee is None else [round(float(l_knee[0]), 2), round(float(l_knee[1]), 2)],
            None if l_ank is None else [round(float(l_ank[0]), 2), round(float(l_ank[1]), 2)],
        )

        missing_keys = [k for k, v in required.items() if v is None]
        if missing_keys:
            logger.error("POSE FAILED – no keypoints detected")
            logger.error("missing keypoints: %s", missing_keys)
            logger.error(
                "measurements None reason: cannot compute body metrics because required keypoints are missing"
            )
            raise RuntimeError(f"POSE_NO_KEYPOINTS: missing {','.join(missing_keys)}")

        shoulder_w = float(np.linalg.norm(l_sh - r_sh))
        hip_w = float(np.linalg.norm(l_hip - r_hip))
        shoulder_mid = (l_sh + r_sh) * 0.5
        hip_mid = (l_hip + r_hip) * 0.5
        l_leg = float(np.linalg.norm(l_hip - l_knee) + np.linalg.norm(l_knee - l_ank))
        r_leg = float(np.linalg.norm(r_hip - r_knee) + np.linalg.norm(r_knee - r_ank))
        leg_h = max((l_leg + r_leg) * 0.5, 1.0)
        torso_h = max(float(np.linalg.norm(shoulder_mid - hip_mid)), 1.0)
        ankle_mid = (l_ank + r_ank) * 0.5

        body_h = max(ankle_mid[1] - nose[1], 1.0)
        shoulder_ratio = shoulder_w / body_h
        hip_ratio = hip_w / body_h
        leg_ratio = leg_h / body_h
        torso_ratio = torso_h / body_h
        total_height_ratio = body_h / max(float(h), 1.0)

        shoulder_ref_px = REF_SHOULDER_RATIO * body_h
        hip_ref_px = REF_HIP_RATIO * body_h
        torso_ref_px = REF_TORSO_RATIO * body_h
        leg_ref_px = REF_LEG_RATIO * body_h
        total_height_ref_px = REF_TOTAL_HEIGHT_RATIO * float(h)

        # Exact formula: scale = userMeasurement / referenceMeasurement
        shoulder_scale_raw = shoulder_w / max(shoulder_ref_px, 1.0)
        hip_scale_raw = hip_w / max(hip_ref_px, 1.0)
        torso_scale_raw = torso_h / max(torso_ref_px, 1.0)
        leg_scale_raw = leg_h / max(leg_ref_px, 1.0)
        height_scale_raw = body_h / max(total_height_ref_px, 1.0)

        shoulder_scale = _clamp(shoulder_scale_raw, 0.82, 1.22)
        hip_scale = _clamp(hip_scale_raw, 0.85, 1.20)
        leg_scale = _clamp(leg_scale_raw, 0.90, 1.15)
        torso_scale = _clamp(torso_scale_raw, 0.90, 1.15)
        waist_scale = _clamp(0.6 * hip_scale + 0.4 * shoulder_scale, 0.86, 1.16)
        chest_scale = _clamp(0.9 * shoulder_scale + 0.1 * waist_scale, 0.86, 1.18)
        height_scale = _clamp(height_scale_raw, 0.93, 1.10)

        scales = {
            "heightScale": float(height_scale),
            "shoulderWidthScale": float(shoulder_scale),
            "chestScale": float(chest_scale),
            "waistScale": float(waist_scale),
            "hipScale": float(hip_scale),
            "legLengthScale": float(leg_scale),
            "torsoLengthScale": float(torso_scale),
        }

        measurements = {
            "shoulderWidthPx": round(shoulder_w, 2),
            "hipWidthPx": round(hip_w, 2),
            "torsoLengthPx": round(torso_h, 2),
            "legLengthPx": round(leg_h, 2),
            "totalHeightPx": round(body_h, 2),
            "totalHeightRatio": round(float(total_height_ratio), 4),
        }
        none_measurements = [k for k, v in measurements.items() if v is None]
        if none_measurements:
            logger.error("measurements None: %s", none_measurements)
            logger.error("measurements None reason: numeric calculation failed")
            raise RuntimeError("POSE_FAILED: invalid measurements (None)")
        references = {
            "shoulderWidthPxRef": round(float(shoulder_ref_px), 2),
            "hipWidthPxRef": round(float(hip_ref_px), 2),
            "torsoLengthPxRef": round(float(torso_ref_px), 2),
            "legLengthPxRef": round(float(leg_ref_px), 2),
            "totalHeightPxRef": round(float(total_height_ref_px), 2),
        }
        formulas = {
            "shoulderWidthScale": "shoulderWidthScale = shoulderWidthPx / shoulderWidthPxRef",
            "hipScale": "hipScale = hipWidthPx / hipWidthPxRef",
            "torsoLengthScale": "torsoLengthScale = torsoLengthPx / torsoLengthPxRef",
            "legLengthScale": "legLengthScale = legLengthPx / legLengthPxRef",
            "heightScale": "heightScale = totalHeightPx / totalHeightPxRef",
        }

        logger.info("pose raw measurements(px): %s", measurements)
        logger.info("pose reference measurements(px): %s", references)
        logger.info("pose scale formulas: %s", formulas)
        logger.info(
            "pose scale raw values: shoulder=%.4f hip=%.4f torso=%.4f leg=%.4f height=%.4f",
            shoulder_scale_raw,
            hip_scale_raw,
            torso_scale_raw,
            leg_scale_raw,
            height_scale_raw,
        )
        logger.info("pose scale clamped values: %s", scales)
        debug_pose = {
            "keypoints_count": int(len(lms)),
            "pose_confidence": float(confidence),
            "person_bbox_px": person_bbox,
            "left_wrist_px": None if l_wrist is None else [round(float(l_wrist[0]), 2), round(float(l_wrist[1]), 2)],
            "right_wrist_px": None if r_wrist is None else [round(float(r_wrist[0]), 2), round(float(r_wrist[1]), 2)],
        }
        return scales, measurements, references, formulas, debug_pose
    except AppError:
        raise
    except Exception as e:
        if str(e).startswith("POSE_FAILED:") or str(e).startswith("POSE_INCOMPLETE_KEYPOINTS:") or str(e).startswith("POSE_NO_KEYPOINTS:"):
            raise
        logger.exception("POSE_FAILED: unexpected body estimation error: %s", e)
        raise RuntimeError("POSE_FAILED: unexpected body estimation error") from e


def _estimate_body_scales(body_front_rgb: np.ndarray, body_side_rgb: Optional[np.ndarray], request_id: Optional[str] = None):
    """
    MVP body fitting from uploaded images.
    Uses front pose landmarks for height/shoulders/waist-hips/leg length.
    Side image is reserved for future depth/chest refinement.
    """
    scales, measurements, references, formulas, debug_pose = _estimate_body_fit_from_pose(body_front_rgb, request_id=request_id)
    return scales, measurements, references, formulas, debug_pose


def get_face_region(rgb: np.ndarray) -> Optional[tuple]:
    detector = _get_face_detector()
    if detector is None:
        return _face_region_opencv(rgb)
    try:
        result = detector.detect(_numpy_to_mp_image(rgb))
        if not result.face_landmarks:
            return _face_region_opencv(rgb)
        h, w = rgb.shape[:2]
        lms = result.face_landmarks[0]
        xs = [lm.x * w for lm in lms]
        ys = [lm.y * h for lm in lms]
        x1 = max(0, min(xs) - 30)
        x2 = min(w, max(xs) + 30)
        y1 = max(0, min(ys) - 30)
        y2 = min(h, max(ys) + 30)
        return (int(x1), int(y1), int(x2), int(y2))
    except Exception:
        return _face_region_opencv(rgb)


def _get_face_landmarks_points(rgb: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    """
    Return FaceMesh landmarks in pixel coordinates [N,2], and source label.
    source: facemesh | fallback | failed
    """
    detector = _get_face_detector()
    if detector is None:
        region = _face_region_opencv(rgb)
        if region is None:
            return None, "failed"
        return None, "fallback"
    try:
        result = detector.detect(_numpy_to_mp_image(rgb))
        if not result.face_landmarks:
            region = _face_region_opencv(rgb)
            return (None, "fallback") if region is not None else (None, "failed")
        h, w = rgb.shape[:2]
        lms = result.face_landmarks[0]
        pts = np.array([[float(lm.x) * w, float(lm.y) * h] for lm in lms], dtype=np.float32)
        # Keep canonical FaceMesh landmark count for stable metric extraction.
        if len(pts) >= 468:
            pts = pts[:468]
        if pts.size == 0:
            return None, "failed"
        return pts, "facemesh"
    except Exception:
        region = _face_region_opencv(rgb)
        return (None, "fallback") if region is not None else (None, "failed")


def get_face_texture(face_img: np.ndarray, size: int = None) -> Optional[np.ndarray]:
    size = size or TEX_SIZE
    rgb = face_img if face_img.ndim == 3 and face_img.shape[-1] == 3 else cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    region = get_face_region(rgb)
    if not region:
        return None
    x1, y1, x2, y2 = region
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (size, size))


def _clamp_material_scalar(value, lo: float, hi: float):
    try:
        return max(lo, min(hi, float(value)))
    except Exception:
        return None


def _material_base_color_factor(mat):
    base = getattr(mat, "baseColorFactor", None)
    if base is not None:
        return base
    color = getattr(mat, "color", None)
    if color is None:
        return None
    try:
        if hasattr(color, "__len__"):
            vals = list(color)
            if len(vals) >= 3:
                return vals[:4] if len(vals) >= 4 else [vals[0], vals[1], vals[2], 1.0]
    except Exception:
        pass
    return None


def _material_emissive_factor(mat):
    em = getattr(mat, "emissiveFactor", None)
    if em is not None:
        return em
    em_color = getattr(mat, "emissive", None)
    if em_color is None:
        return None
    try:
        if hasattr(em_color, "__len__"):
            vals = list(em_color)
            if len(vals) >= 3:
                return vals[:3]
    except Exception:
        pass
    return None


def _is_near_white_rgb(values) -> bool:
    try:
        r, g, b = float(values[0]), float(values[1]), float(values[2])
        # Support both normalized [0..1] and byte [0..255] inputs.
        # Some trimesh material types expose colors as uint8 arrays.
        if r > 1.5 or g > 1.5 or b > 1.5:
            r, g, b = r / 255.0, g / 255.0, b / 255.0
        return r > 0.94 and g > 0.94 and b > 0.94
    except Exception:
        return False


def _set_mid_gray_base_color_if_needed(mat) -> None:
    """
    Keep existing material object, but avoid washed-out near-white base color.
    If base color is near-white on all channels, set it to mid-gray.
    """
    base = getattr(mat, "baseColorFactor", None)
    if base is not None and _is_near_white_rgb(base):
        try:
            alpha = float(base[3]) if len(base) > 3 else 1.0
        except Exception:
            alpha = 1.0
        mat.baseColorFactor = [0.6, 0.6, 0.6, alpha]
        return

    color = getattr(mat, "color", None)
    if color is None:
        return
    try:
        if hasattr(color, "__len__"):
            vals = list(color)
            if len(vals) >= 3 and _is_near_white_rgb(vals):
                mat.color = [0.6, 0.6, 0.6]
    except Exception:
        pass


def _set_black_emissive_if_present(mat) -> None:
    if hasattr(mat, "emissiveFactor"):
        try:
            mat.emissiveFactor = [0.0, 0.0, 0.0]
        except Exception:
            pass
    if hasattr(mat, "emissive"):
        try:
            em = getattr(mat, "emissive")
            if hasattr(em, "fill"):
                em.fill(0.0)
            elif hasattr(em, "__setitem__"):
                em[0], em[1], em[2] = 0.0, 0.0, 0.0
            else:
                mat.emissive = [0.0, 0.0, 0.0]
        except Exception:
            pass


def _material_props(mat) -> dict:
    return {
        "name": getattr(mat, "name", None),
        "alphaMode": getattr(mat, "alphaMode", None),
        "transparent": getattr(mat, "transparent", None),
        "opacity": getattr(mat, "opacity", None),
        "baseColorFactor": _material_base_color_factor(mat),
        "emissiveFactor": _material_emissive_factor(mat),
        "roughnessFactor": getattr(mat, "roughnessFactor", getattr(mat, "roughness", None)),
        "metallicFactor": getattr(mat, "metallicFactor", getattr(mat, "metalness", None)),
    }


def _iter_unique_scene_materials(scene: trimesh.Scene):
    seen = set()
    idx = 0
    for g in scene.geometry.values():
        visual = getattr(g, "visual", None)
        mat = getattr(visual, "material", None) if visual is not None else None
        if mat is None:
            continue
        key = id(mat)
        if key in seen:
            continue
        seen.add(key)
        yield idx, mat
        idx += 1


def _create_fallback_material():
    """Create a safe opaque mid-gray fallback material only when missing."""
    if PBRMaterial is not None:
        return PBRMaterial(
            baseColorFactor=[0.6, 0.6, 0.6, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.8,
            alphaMode="OPAQUE",
            emissiveFactor=[0.0, 0.0, 0.0],
        )
    return SimpleMaterial(diffuse=[153, 153, 153, 255], ambient=[153, 153, 153, 255], glossiness=0.0)


def _ensure_geometry_materials(scene: trimesh.Scene) -> None:
    """
    Preserve original mannequin materials.
    Create a new material only when a geometry has none.
    """
    for name, g in scene.geometry.items():
        visual = getattr(g, "visual", None)
        if visual is None:
            continue
        mat = getattr(visual, "material", None)
        if mat is None:
            try:
                visual.material = _create_fallback_material()
                logger.info("material created for geometry '%s' (missing original)", name)
            except Exception as e:
                logger.warning("Could not assign fallback material for geometry '%s': %s", name, e)


def _sanitize_material_in_place(mat, material_idx: int) -> None:
    """
    Keep original material object, but enforce safe opaque PBR ranges.
    No material replacement or recreation.
    """
    if mat is None:
        logger.info("material[%s]: <none>", material_idx)
        return

    before = _material_props(mat)
    logger.info("material[%s] before: %s", material_idx, before)

    # Force opaque rendering properties, without replacing the material.
    if hasattr(mat, "transparent"):
        mat.transparent = False
    if hasattr(mat, "opacity"):
        mat.opacity = 1.0
    if hasattr(mat, "alphaMode"):
        mat.alphaMode = "OPAQUE"
    if hasattr(mat, "alphaMap"):
        mat.alphaMap = None

    # Avoid washed-out base colors.
    _set_mid_gray_base_color_if_needed(mat)

    # Ensure emissive does not wash out shading.
    _set_black_emissive_if_present(mat)

    # Keep physically plausible ranges.
    if hasattr(mat, "metallicFactor"):
        clamped = _clamp_material_scalar(mat.metallicFactor, 0.0, 0.2)
        if clamped is not None:
            mat.metallicFactor = clamped
    if hasattr(mat, "metalness"):
        clamped = _clamp_material_scalar(mat.metalness, 0.0, 0.2)
        if clamped is not None:
            mat.metalness = clamped
    if hasattr(mat, "roughnessFactor"):
        clamped = _clamp_material_scalar(mat.roughnessFactor, 0.6, 0.9)
        if clamped is not None:
            mat.roughnessFactor = clamped
    if hasattr(mat, "roughness"):
        clamped = _clamp_material_scalar(mat.roughness, 0.6, 0.9)
        if clamped is not None:
            mat.roughness = clamped

    after = _material_props(mat)
    logger.info("material[%s] after: %s", material_idx, after)


def _log_and_sanitize_scene_materials(scene: trimesh.Scene) -> None:
    """Log material properties and enforce clean opaque output before export."""
    _ensure_geometry_materials(scene)
    for idx, mat in _iter_unique_scene_materials(scene):
        _sanitize_material_in_place(mat, idx)


def _log_exported_scene_materials(scene: trimesh.Scene) -> None:
    """Log final material properties from exported/reloaded scene."""
    for idx, mat in _iter_unique_scene_materials(scene):
        p = _material_props(mat)
        logger.info(
            "exported material[%s]: name=%s baseColorFactor=%s metallicFactor=%s roughnessFactor=%s alphaMode=%s",
            idx,
            p.get("name"),
            p.get("baseColorFactor"),
            p.get("metallicFactor"),
            p.get("roughnessFactor"),
            p.get("alphaMode"),
        )


def _exported_materials_summary(scene: trimesh.Scene, top_n: int = 8) -> list[dict]:
    """
    JSON-safe summary of exported materials (most-used first when possible).
    This is used for debugging when users report that skin tone didn't change.
    """
    try:
        return _material_debug_summary(scene, top_n=top_n)
    except Exception:
        return []


def _skin_pixels_from_patch_rgb(patch_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Return Nx3 RGB pixels likely to be skin, or None if insufficient.
    Uses a lightweight YCrCb threshold mask (no auto lightening/darkening).
    """
    try:
        if patch_rgb is None or patch_rgb.size == 0:
            return None
        if patch_rgb.dtype != np.uint8:
            patch_rgb = np.clip(patch_rgb, 0, 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2YCrCb)
        # OpenCV order: Y, Cr, Cb
        Y = ycrcb[:, :, 0]
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]
        mask = (Y > 40) & (Cr >= 135) & (Cr <= 180) & (Cb >= 85) & (Cb <= 135)
        pixels = patch_rgb[mask]
        if pixels is None or len(pixels) < 60:
            return None
        return pixels.reshape(-1, 3)
    except Exception:
        return None


def _patch_from_bbox(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    try:
        h, w = rgb.shape[:2]
        x1 = int(max(0, min(w - 1, x1)))
        x2 = int(max(0, min(w, x2)))
        y1 = int(max(0, min(h - 1, y1)))
        y2 = int(max(0, min(h, y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        patch = rgb[y1:y2, x1:x2]
        return patch if patch is not None and patch.size > 0 else None
    except Exception:
        return None


def _estimate_skin_rgb(
    face_rgb: np.ndarray,
    body_rgb: np.ndarray,
    pose_debug: Optional[dict] = None,
    request_id: Optional[str] = None,
) -> tuple[int, int, int]:
    """
    Lightweight skin-tone estimate:
    - Prefer sampling from cheek regions within detected face bbox.
    - Also sample from hand regions using pose wrist keypoints when available.
    - No auto lightening/darkening; preserve undertones by using median RGB.
    - If skin detection fails: fallback to average from cheek area (no mask).
    """
    try:
        sampled_pixels = []
        fallback_cheek_patch = None

        # 1) Face cheeks
        region = get_face_region(face_rgb)
        if region:
            fx1, fy1, fx2, fy2 = region
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)
            # Cheek rectangles (approx): lower-mid of face bbox, left/right.
            left_cheek = (
                int(fx1 + 0.15 * fw),
                int(fy1 + 0.35 * fh),
                int(fx1 + 0.40 * fw),
                int(fy1 + 0.65 * fh),
            )
            right_cheek = (
                int(fx1 + 0.60 * fw),
                int(fy1 + 0.35 * fh),
                int(fx1 + 0.85 * fw),
                int(fy1 + 0.65 * fh),
            )
            cheek_patches = [
                _patch_from_bbox(face_rgb, *left_cheek),
                _patch_from_bbox(face_rgb, *right_cheek),
            ]
            # For fallback: use combined cheek region
            fallback_cheek_patch = _patch_from_bbox(
                face_rgb,
                int(fx1 + 0.18 * fw),
                int(fy1 + 0.35 * fh),
                int(fx1 + 0.82 * fw),
                int(fy1 + 0.68 * fh),
            )
            for p in cheek_patches:
                skin_px = _skin_pixels_from_patch_rgb(p) if p is not None else None
                if skin_px is not None:
                    sampled_pixels.append(skin_px)

        # 2) Hands/wrists (from pose)
        if pose_debug and body_rgb is not None:
            h, w = body_rgb.shape[:2]
            wrist_pts = []
            lw = pose_debug.get("left_wrist_px")
            rw = pose_debug.get("right_wrist_px")
            if isinstance(lw, (list, tuple)) and len(lw) == 2:
                wrist_pts.append((float(lw[0]), float(lw[1])))
            if isinstance(rw, (list, tuple)) and len(rw) == 2:
                wrist_pts.append((float(rw[0]), float(rw[1])))
            patch_r = int(max(10, 0.06 * min(h, w)))
            for (px, py) in wrist_pts:
                x1, y1 = int(px - patch_r), int(py - patch_r)
                x2, y2 = int(px + patch_r), int(py + patch_r)
                patch = _patch_from_bbox(body_rgb, x1, y1, x2, y2)
                skin_px = _skin_pixels_from_patch_rgb(patch) if patch is not None else None
                if skin_px is not None:
                    sampled_pixels.append(skin_px)

        if sampled_pixels:
            all_px = np.vstack(sampled_pixels)
            med = np.median(all_px, axis=0)
            rgb = (int(med[0]), int(med[1]), int(med[2]))
            logger.info(
                "skin sampling: request_id=%s sources=%s pixels=%s rgb=%s",
                request_id,
                ["face_cheek", "hands_wrist"],
                int(all_px.shape[0]),
                rgb,
            )
            return rgb

        # 3) If skin detection fails, sample average from cheek area (no mask)
        if fallback_cheek_patch is not None and fallback_cheek_patch.size > 0:
            med = np.median(fallback_cheek_patch.reshape(-1, 3), axis=0)
            rgb = (int(med[0]), int(med[1]), int(med[2]))
            logger.info(
                "skin sampling fallback (cheek area, no mask): request_id=%s rgb=%s",
                request_id,
                rgb,
            )
            return rgb

        # 4) Last resort: upper-center body patch (no generic placeholder)
        if body_rgb is not None:
            h, w = body_rgb.shape[:2]
            y1, y2 = int(h * 0.15), int(h * 0.45)
            x1, x2 = int(w * 0.35), int(w * 0.65)
            patch = _patch_from_bbox(body_rgb, x1, y1, x2, y2)
            if patch is not None and patch.size > 0:
                med = np.median(patch.reshape(-1, 3), axis=0)
                rgb = (int(med[0]), int(med[1]), int(med[2]))
                logger.info("skin sampling fallback (upper body patch): request_id=%s rgb=%s", request_id, rgb)
                return rgb

        return (176, 146, 120)
    except Exception:
        return (176, 146, 120)


def _apply_skin_basecolor(scene: trimesh.Scene, skin_rgb: tuple[int, int, int], request_id: Optional[str] = None) -> dict:
    """
    Apply skin baseColorFactor (linear) to likely skin materials (body/head),
    without replacing existing textures/materials.
    """
    target_mats = _select_body_materials(scene)
    linear_factor = _rgb8_to_linear_factor(skin_rgb)
    updated = []
    touched = 0
    for mat in target_mats:
        try:
            if hasattr(mat, "baseColorFactor"):
                mat.baseColorFactor = list(linear_factor)
                # Skin should not be metallic; keep it matte-ish.
                if hasattr(mat, "metallicFactor"):
                    mat.metallicFactor = 0.0
                if hasattr(mat, "metalness"):
                    mat.metalness = 0.0
                if hasattr(mat, "roughnessFactor"):
                    mat.roughnessFactor = 0.85
                if hasattr(mat, "roughness"):
                    mat.roughness = 0.85
                if hasattr(mat, "alphaMode"):
                    mat.alphaMode = "OPAQUE"
                if hasattr(mat, "transparent"):
                    mat.transparent = False
                if hasattr(mat, "opacity"):
                    mat.opacity = 1.0
                touched += 1
                updated.append(_material_name(mat))
            elif hasattr(mat, "color"):
                # Fallback for simple materials (RGBA 0-255)
                mat.color = [int(skin_rgb[0]), int(skin_rgb[1]), int(skin_rgb[2]), 255]
                touched += 1
                updated.append(_material_name(mat))
        except Exception:
            continue
    logger.info(
        "skin baseColor applied: request_id=%s rgb=%s linear=%s materials=%s touched=%s",
        request_id,
        skin_rgb,
        [round(float(x), 4) for x in linear_factor[:3]],
        updated,
        touched,
    )
    return {"rgb8": list(skin_rgb), "linear": linear_factor, "materials": updated, "touched": int(touched)}


def _apply_face_texture(scene: trimesh.Scene, face_rgb: np.ndarray, request_id: Optional[str] = None) -> dict:
    """
    Try to apply a face-crop texture to head/face materials if possible.
    This preserves original materials (only sets baseColorTexture/image on targeted mats).
    """
    updated = []
    try:
        tex = get_face_texture(face_rgb, size=512)
        if tex is None or tex.size == 0:
            logger.info("face texture: request_id=%s face_crop_not_found", request_id)
            return {"applied": False, "materials": []}
        tex_img = Image.fromarray(tex.astype(np.uint8), mode="RGB")
        head_mats = _select_head_materials(scene)
        if not head_mats:
            logger.info("face texture: request_id=%s no_head_materials_found", request_id)
            return {"applied": False, "materials": []}
        for mat in head_mats:
            try:
                if hasattr(mat, "baseColorTexture"):
                    mat.baseColorTexture = tex_img
                # Many trimesh materials use .image as base-color image
                if hasattr(mat, "image"):
                    mat.image = tex_img
                updated.append(_material_name(mat))
            except Exception:
                continue
        logger.info("face texture applied: request_id=%s materials=%s", request_id, updated)
        return {"applied": bool(updated), "materials": updated}
    except Exception as e:
        logger.warning("face texture apply failed request_id=%s err=%s", request_id, e)
        return {"applied": False, "materials": []}


def _apply_debug_visual_deform(scene: trimesh.Scene, scales: dict) -> tuple[dict, dict, list[str]]:
    """
    Debug deformation path:
    1) clamp scales and log
    2) try node/bone-like transforms
    3) fallback to per-vertex deformation for guaranteed visible effect
    """
    warnings = []
    before = _scene_bbox(scene)
    raw = dict(scales)
    clamped = dict(scales)
    for k in ("heightScale", "shoulderWidthScale", "hipScale", "legLengthScale", "torsoLengthScale", "chestScale", "waistScale"):
        clamped[k] = _clamp(float(clamped.get(k, 1.0)), 0.85, 1.20)
    logger.info("debug deform scales raw=%s clamped=%s", raw, clamped)

    def _scale_node_if_match(node_name: str, sx=1.0, sy=1.0, sz=1.0) -> bool:
        low = node_name.lower()
        transform = None
        for n in scene.graph.nodes:
            if node_name in str(n).lower() or low in str(n).lower():
                transform, _ = scene.graph.get(n)
                m = np.array(transform, dtype=np.float64)
                m[:3, :3] = m[:3, :3] @ np.diag([sx, sy, sz])
                scene.graph.update(frame_to=n, matrix=m)
                return True
        return False

    node_hits = 0
    node_hits += int(_scale_node_if_match("root", sy=clamped["heightScale"]))
    node_hits += int(_scale_node_if_match("spine", sx=clamped["shoulderWidthScale"]))
    node_hits += int(_scale_node_if_match("pelvis", sx=clamped["hipScale"]))
    node_hits += int(_scale_node_if_match("thigh", sy=clamped["legLengthScale"]))

    # Guaranteed fallback: vertex-level deformation (visible regardless of rig support).
    for g in scene.geometry.values():
        if not hasattr(g, "vertices") or len(g.vertices) == 0:
            continue
        v = np.array(g.vertices, dtype=np.float64, copy=True)
        y = v[:, 1]
        y_min, y_max = float(np.min(y)), float(np.max(y))
        span = max(y_max - y_min, 1e-6)
        t = (y - y_min) / span

        shoulder_band = np.exp(-0.5 * ((t - 0.78) / 0.09) ** 2)
        hip_band = np.exp(-0.5 * ((t - 0.48) / 0.10) ** 2)
        sx = 1.0 + (clamped["shoulderWidthScale"] - 1.0) * shoulder_band + (clamped["hipScale"] - 1.0) * hip_band
        cx = float((g.bounds[0][0] + g.bounds[1][0]) * 0.5)
        v[:, 0] = cx + (v[:, 0] - cx) * np.clip(sx, 0.7, 1.4)

        split_y = y_min + span * 0.52
        leg_mask = v[:, 1] < split_y
        v[leg_mask, 1] = split_y + (v[leg_mask, 1] - split_y) * clamped["legLengthScale"]

        torso_lo, torso_hi = y_min + span * 0.52, y_min + span * 0.84
        torso_mask = (v[:, 1] >= torso_lo) & (v[:, 1] <= torso_hi)
        v[torso_mask, 1] = torso_lo + (v[torso_mask, 1] - torso_lo) * clamped["torsoLengthScale"]

        v[:, 1] = y_min + (v[:, 1] - y_min) * clamped["heightScale"]
        g.vertices = v

    after = _scene_bbox(scene)
    logger.info("debug deform bbox before=%s after=%s node_hits=%s", before, after, node_hits)
    if all(abs(float(clamped.get(k, 1.0)) - 1.0) < 1e-6 for k in BODY_SCALE_DEFAULTS.keys()):
        warnings.append("no_effective_personalization")
    if node_hits == 0:
        warnings.append("node_scaling_not_available_used_vertex_fallback")
    return before, after, warnings


def _apply_extreme_debug_deform(scene: trimesh.Scene) -> tuple[dict, dict, int]:
    """
    Extreme debug deformation to prove mesh is changed before export.
    - vertex.y *= 1.3
    - vertex.x *= 0.7
    """
    before = _scene_bbox(scene)
    vertex_count = 0
    for g in scene.geometry.values():
        if not hasattr(g, "vertices") or len(g.vertices) == 0:
            continue
        v = np.array(g.vertices, dtype=np.float64, copy=True)
        vertex_count += len(v)
        v[:, 1] *= 1.3
        v[:, 0] *= 0.7
        g.vertices = v
    after = _scene_bbox(scene)
    logger.warning("EXTREME DEBUG DEFORM APPLIED: bbox BEFORE=%s AFTER=%s vertex_count=%s", before, after, vertex_count)
    return before, after, vertex_count


def _apply_body_proportions(mesh: trimesh.Trimesh, scales: dict) -> None:
    """Apply lightweight segment-based deformation on X/Z widths and leg/height on Y."""
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        return
    verts = np.array(mesh.vertices, dtype=np.float64, copy=True)
    y = verts[:, 1]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = max(y_max - y_min, 1e-6)
    t = (y - y_min) / span

    shoulder_s = float(scales.get("shoulderWidthScale", 1.0))
    chest_s = float(scales.get("chestScale", 1.0))
    waist_s = float(scales.get("waistScale", 1.0))
    hip_s = float(scales.get("hipScale", 1.0))
    leg_s = float(scales.get("legLengthScale", 1.0))
    torso_s = float(scales.get("torsoLengthScale", 1.0))
    height_s = float(scales.get("heightScale", 1.0))

    # Smooth vertical weighting for body segments in normalized Y.
    w_sh = np.exp(-0.5 * ((t - 0.78) / 0.08) ** 2)
    w_ch = np.exp(-0.5 * ((t - 0.67) / 0.09) ** 2)
    w_wa = np.exp(-0.5 * ((t - 0.58) / 0.08) ** 2)
    w_hip = np.exp(-0.5 * ((t - 0.48) / 0.09) ** 2)

    sx = 1.0 + (shoulder_s - 1.0) * w_sh + (waist_s - 1.0) * w_wa + (hip_s - 1.0) * w_hip
    sz = 1.0 + (chest_s - 1.0) * w_ch + (waist_s - 1.0) * w_wa + (hip_s - 1.0) * w_hip
    sx = np.clip(sx, 0.75, 1.35)
    sz = np.clip(sz, 0.75, 1.35)

    bounds = mesh.bounds
    cx = float((bounds[0][0] + bounds[1][0]) * 0.5)
    cz = float((bounds[0][2] + bounds[1][2]) * 0.5)
    verts[:, 0] = cx + (verts[:, 0] - cx) * sx
    verts[:, 2] = cz + (verts[:, 2] - cz) * sz

    # Leg-length deformation from split down to feet.
    split_y = y_min + span * 0.52
    below = verts[:, 1] < split_y
    verts[below, 1] = split_y + (verts[below, 1] - split_y) * leg_s

    # Torso-length deformation between hips and upper torso (keeps feet/head stable-ish).
    torso_lo = y_min + span * 0.52
    torso_hi = y_min + span * 0.82
    torso_mask = (verts[:, 1] >= torso_lo) & (verts[:, 1] <= torso_hi)
    verts[torso_mask, 1] = torso_lo + (verts[torso_mask, 1] - torso_lo) * torso_s

    # Global vertical scale from feet anchor.
    verts[:, 1] = y_min + (verts[:, 1] - y_min) * height_s

    mesh.vertices = verts


def _log_body_node_transforms(scene: trimesh.Scene) -> None:
    """
    Log scene graph transforms for torso/hip/leg-related nodes before export.
    Useful to verify whether scaling happened via node/bone transforms or mesh deformation.
    """
    keywords = ("torso", "spine", "pelvis", "hip", "thigh", "calf", "leg", "knee")
    matched = 0
    for node in scene.graph.nodes:
        node_name = str(node)
        low = node_name.lower()
        if not any(k in low for k in keywords):
            continue
        try:
            transform, geom_name = scene.graph.get(node_name)
            t = np.array(transform, dtype=np.float64)
            pos = [round(float(v), 4) for v in t[:3, 3]]
            # Approximate per-axis scale from transform columns.
            sc = [round(float(v), 4) for v in np.linalg.norm(t[:3, :3], axis=0)]
            logger.info(
                "node transform: node=%s geometry=%s position=%s scale=%s",
                node_name,
                geom_name,
                pos,
                sc,
            )
            matched += 1
        except Exception as e:
            logger.warning("Could not read node transform for '%s': %s", node_name, e)
    if matched == 0:
        logger.warning("No torso/hip/leg nodes found for transform logging.")


def _ensure_base_mannequin(avatar_style: Optional[str] = None) -> Optional[Path]:
    """Ensure base human model exists. No mannequin auto-download in human mode."""
    return _get_base_mannequin_path(avatar_style)


def _bbox_from_geometries(geoms: list) -> dict:
    bounds = []
    for g in geoms:
        if hasattr(g, "bounds"):
            try:
                bounds.append(np.asarray(g.bounds, dtype=float).reshape(2, 3))
            except Exception:
                continue
    if not bounds:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    all_bounds = np.vstack(bounds)  # shape (2N, 3)
    bbox_min = all_bounds.min(axis=0)
    bbox_max = all_bounds.max(axis=0)
    size = np.asarray(bbox_max - bbox_min, dtype=float).reshape(-1)
    return {
        "x": float(size[0]),
        "y": float(size[1]),
        "z": float(size[2]),
        "min": [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])],
        "max": [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])],
    }


def _apply_matrix_to_geometries(geoms: list, matrix: np.ndarray) -> None:
    for g in geoms:
        if hasattr(g, "apply_transform"):
            try:
                g.apply_transform(matrix)
            except Exception:
                continue


def _bake_scene_to_world_geometries(loaded_scene: trimesh.Scene) -> tuple[list, dict]:
    """
    Bake node transforms into geometry vertices so exported scene can use identity root.
    """
    out = []
    nodes_with_geometry = 0
    non_identity_nodes = 0
    try:
        node_iter = list(getattr(loaded_scene.graph, "nodes_geometry", []))
    except Exception:
        node_iter = []
    for node_name in node_iter:
        try:
            transform, geom_name = loaded_scene.graph.get(node_name)
            geom = loaded_scene.geometry.get(geom_name)
            if geom is None:
                continue
            g = geom.copy()
            m = np.asarray(transform, dtype=float)
            if m.shape == (4, 4):
                nodes_with_geometry += 1
                if not np.allclose(m, np.eye(4), atol=1e-6):
                    non_identity_nodes += 1
                g.apply_transform(m)
            out.append(g)
        except Exception:
            continue
    if not out:
        out = [g.copy() for g in loaded_scene.geometry.values() if hasattr(g, "vertices")]
    return out, {
        "nodesWithGeometry": int(nodes_with_geometry),
        "nonIdentityNodeTransforms": int(non_identity_nodes),
    }


def _normalize_human_export_geometries(geoms: list, target_human_height_m: float = 1.7) -> tuple[list, dict]:
    """
    Normalize orientation/scale/translation for robust Y-up export with identity root transform.
    """
    dbg = {
        "bboxBefore": _bbox_from_geometries(geoms),
        "bboxAfter": None,
        "axisFixApplied": False,
        "scaleFixApplied": False,
        "scaleFactor": 1.0,
    }
    before = dbg["bboxBefore"]

    # If model is Z-up (height mostly on Z), rotate to Y-up: y' = z, z' = -y.
    if float(before.get("z", 0.0)) > max(float(before.get("y", 0.0)) * 1.20, float(before.get("x", 0.0)) * 1.05):
        z_up_to_y_up = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        _apply_matrix_to_geometries(geoms, z_up_to_y_up)
        dbg["axisFixApplied"] = True

    mid = _bbox_from_geometries(geoms)
    h = float(mid.get("y", 0.0))
    # Safety: absurdly large/small human height -> normalize to target.
    if h > 10.0 or (h > 1e-6 and h < 0.5):
        sf = float(target_human_height_m) / max(h, 1e-6)
        sm = np.array(
            [
                [sf, 0.0, 0.0, 0.0],
                [0.0, sf, 0.0, 0.0],
                [0.0, 0.0, sf, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        _apply_matrix_to_geometries(geoms, sm)
        dbg["scaleFixApplied"] = True
        dbg["scaleFactor"] = float(sf)

    # Ground to y=0 and center on X/Z so root transform can remain identity.
    b = _bbox_from_geometries(geoms)
    mn = b.get("min", [0.0, 0.0, 0.0])
    mx = b.get("max", [0.0, 0.0, 0.0])
    tx = -0.5 * (float(mn[0]) + float(mx[0]))
    ty = -float(mn[1])
    tz = -0.5 * (float(mn[2]) + float(mx[2]))
    tm = np.array(
        [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    _apply_matrix_to_geometries(geoms, tm)
    dbg["bboxAfter"] = _bbox_from_geometries(geoms)
    return geoms, dbg


def _load_base_mannequin_scene(
    body_scales: Optional[dict],
    avatar_style: Optional[str] = None,
    allow_vertex_deform: bool = True,
) -> Optional[trimesh.Scene]:
    """Load base mannequin GLB/GLTF and apply geometry-only body scaling."""
    candidate_paths = _get_base_mannequin_candidates(avatar_style)
    ensure_path = _ensure_base_mannequin(avatar_style)
    if ensure_path is not None and all(str(p) != str(ensure_path) for p in candidate_paths):
        candidate_paths.append(ensure_path)
    if not candidate_paths:
        return None
    for path in candidate_paths:
        try:
            loaded = trimesh.load(str(path), force="scene")
            root_transform_before = {"nodesWithGeometry": 0, "nonIdentityNodeTransforms": 0}
            if isinstance(loaded, trimesh.Scene):
                geoms, root_transform_before = _bake_scene_to_world_geometries(loaded)
            else:
                geoms = [loaded.copy()] if hasattr(loaded, "vertices") else []
            if not geoms:
                continue
            geoms_original = [g.copy() for g in geoms if hasattr(g, "vertices")]
            # Human assets from DCC tools often include helper primitives (e.g. a large cube)
            # and accidental duplicated meshes. Clean these before scaling/export.
            geoms = _clean_human_base_geometries(geoms)
            if not geoms:
                logger.warning("human base clean removed all geometry; falling back to original meshes (%s)", path.name)
                geoms = geoms_original
            if not geoms:
                continue
            scales = {**BODY_SCALE_DEFAULTS, **(body_scales or {})}
            # Human bases have different topology/orientation than mannequin.
            # Mannequin-specific deformation can twist/flip them, so skip it.
            is_human_base = str(path.name).lower().startswith("base_human")
            if not is_human_base and allow_vertex_deform:
                for g in geoms:
                    if not hasattr(g, "vertices"):
                        continue
                    _apply_body_proportions(g, scales)
                export_norm_debug = {
                    "bboxBefore": _bbox_from_geometries(geoms),
                    "bboxAfter": _bbox_from_geometries(geoms),
                    "axisFixApplied": False,
                    "scaleFixApplied": False,
                    "scaleFactor": 1.0,
                }
            elif not is_human_base and not allow_vertex_deform:
                logger.warning("mannequin deformation skipped: morph targets missing on base model (%s)", path.name)
                export_norm_debug = {
                    "bboxBefore": _bbox_from_geometries(geoms),
                    "bboxAfter": _bbox_from_geometries(geoms),
                    "axisFixApplied": False,
                    "scaleFixApplied": False,
                    "scaleFactor": 1.0,
                    "deformationSkipped": True,
                    "deformationSkipReason": "morphs_missing",
                }
            else:
                try:
                    geoms, export_norm_debug = _normalize_human_export_geometries(geoms, target_human_height_m=1.7)
                except Exception as norm_err:
                    logger.warning("export normalization failed (%s), continuing without it: %s", path.name, norm_err)
                    geoms = geoms_original if geoms_original else geoms
                    export_norm_debug = {
                        "bboxBefore": _bbox_from_geometries(geoms),
                        "bboxAfter": _bbox_from_geometries(geoms),
                        "axisFixApplied": False,
                        "scaleFixApplied": False,
                        "scaleFactor": 1.0,
                        "normalizationError": str(norm_err),
                    }
                logger.info("human base detected -> skipping mannequin deformation path (%s)", path.name)
            scene = trimesh.Scene()
            scene.add_geometry(geoms)
            scene.metadata = dict(getattr(scene, "metadata", {}) or {})
            scene.metadata["basePathUsed"] = str(path.resolve())
            scene.metadata["rootTransformBefore"] = root_transform_before
            scene.metadata["rootTransformAfter"] = {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            }
            scene.metadata["exportNormalization"] = export_norm_debug
            return scene
        except Exception as e:
            logger.warning("Could not load base mannequin candidate (%s): %s", path.name, e)
            # Last-chance fallback for each candidate: direct geometry load.
            try:
                loaded2 = trimesh.load(str(path), force="scene")
                if isinstance(loaded2, trimesh.Scene):
                    geoms2 = [g.copy() for g in loaded2.geometry.values() if hasattr(g, "vertices")]
                else:
                    geoms2 = [loaded2.copy()] if hasattr(loaded2, "vertices") else []
                if not geoms2:
                    continue
                scene2 = trimesh.Scene()
                scene2.add_geometry(geoms2)
                scene2.metadata = dict(getattr(scene2, "metadata", {}) or {})
                scene2.metadata["basePathUsed"] = str(path.resolve())
                scene2.metadata["rootTransformBefore"] = {"fallbackDirectLoad": True}
                scene2.metadata["rootTransformAfter"] = {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                }
                scene2.metadata["exportNormalization"] = {
                    "bboxBefore": _bbox_from_geometries(geoms2),
                    "bboxAfter": _bbox_from_geometries(geoms2),
                    "axisFixApplied": False,
                    "scaleFixApplied": False,
                    "scaleFactor": 1.0,
                    "fallbackDirectLoad": True,
                }
                logger.warning("base mannequin fallback direct load succeeded: %s", path.name)
                return scene2
            except Exception as e2:
                logger.warning("base mannequin fallback direct load failed (%s): %s", path.name, e2)
                continue
    return None


def _clean_human_base_geometries(geoms: list) -> list:
    """
    Remove helper primitives and duplicate meshes from imported human base assets.
    This fixes cases where a large white cube appears beneath the character.
    """
    cleaned = []
    seen = set()
    removed_helpers = 0
    removed_dupes = 0
    for g in geoms:
        if not hasattr(g, "vertices") or not hasattr(g, "faces"):
            continue
        try:
            v = np.asarray(g.vertices)
            f = np.asarray(g.faces)
            if v.size == 0 or f.size == 0:
                continue
            size = (g.bounds[1] - g.bounds[0]).astype(float)
            mat = getattr(getattr(g, "visual", None), "material", None)
            mat_name = _material_name(mat).lower() if mat is not None else ""
            underwear_kws = ("underwear", "bikini", "bra", "panty", "brief", "boxer", "shorts", "swim")
            if any(k in mat_name for k in underwear_kws):
                cleaned.append(g)
                continue
            # Helper cube heuristic: very low-poly box-like mesh with generic material.
            looks_like_helper_cube = (
                int(len(f)) <= 24
                and int(len(v)) <= 32
                and float(max(size)) >= 1.5
                and float(min(size)) >= 1.5
                and ("material" in mat_name or mat_name in {"", "none"})
            )
            if looks_like_helper_cube:
                removed_helpers += 1
                continue

            # Deduplicate exact overlapping geometry exported twice.
            sig = hashlib.sha1(v.tobytes() + b"|" + f.tobytes()).hexdigest()
            if sig in seen:
                removed_dupes += 1
                continue
            seen.add(sig)
            cleaned.append(g)
        except Exception:
            cleaned.append(g)
    logger.info(
        "human base geometry clean: kept=%s removed_helpers=%s removed_duplicates=%s",
        len(cleaned),
        removed_helpers,
        removed_dupes,
    )
    return cleaned


def get_body_scale_defaults() -> dict:
    """Return default proportion scales used by body fitting."""
    return dict(BODY_SCALE_DEFAULTS)


def debug_mannequin_morphs(avatar_style: Optional[str] = None) -> dict:
    """
    Debug helper for base mannequin morphology and underwear presence.
    """
    path = _get_base_mannequin_path(avatar_style)
    if path is None:
        return {
            "ok": False,
            "path": None,
            "exists": False,
            "hasMorphTargets": False,
            "meshCount": 0,
            "meshesWithMorphTargets": 0,
            "totalMorphTargets": 0,
            "meshMorphCounts": {},
            "meshMorphNames": {},
            "hasUnderwearMesh": False,
            "underwearMeshes": [],
            "error": "base_model_missing",
            "avatarStyle": avatar_style or "neutral",
        }
    out = gltf_debug_morphs(str(path))
    out["avatarStyle"] = avatar_style or "neutral"
    return out


def run_pipeline(
    face_path: str,
    body_front_path: str,
    body_side_path: Optional[str],
    out_glb_path: str,
    progress_cb=None,
    body_scales: Optional[dict] = None,
    request_id: Optional[str] = None,
    avatar_style: Optional[str] = None,
) -> dict:
    def prog(pct, msg=""):
        if progress_cb:
            progress_cb(pct, msg)
        logger.info("pipeline progress %s: %s", pct, msg)

    prog(5, "Loading images")
    # B) Compute input hashes to prove we use the correct files (no caching).
    face_raw = None
    body_front_raw = None
    body_side_raw = None
    try:
        face_raw = Path(face_path).read_bytes()
        body_front_raw = Path(body_front_path).read_bytes()
        if body_side_path:
            body_side_raw = Path(body_side_path).read_bytes()
    except Exception:
        pass
    face_image_hash = hashlib.sha256(face_raw).hexdigest()[:16] if face_raw else None
    body_front_image_hash = hashlib.sha256(body_front_raw).hexdigest()[:16] if body_front_raw else None
    body_side_image_hash = hashlib.sha256(body_side_raw).hexdigest()[:16] if body_side_raw else None
    face_file_size = len(face_raw) if face_raw else -1
    body_front_file_size = len(body_front_raw) if body_front_raw else -1
    logger.info(
        "input hashes (B) request_id=%s face_hash=%s face_bytes=%s body_front_hash=%s body_front_bytes=%s body_side_hash=%s body_side_bytes=%s face_path=%s body_path=%s body_side_path=%s",
        request_id,
        face_image_hash,
        face_file_size,
        body_front_image_hash,
        body_front_file_size,
        body_side_image_hash,
        len(body_side_raw) if body_side_raw else -1,
        str(Path(face_path).resolve()),
        str(Path(body_front_path).resolve()),
        str(Path(body_side_path).resolve()) if body_side_path else None,
    )
    face_img = _read_image(face_path)
    body_img = _read_image(body_front_path)
    body_side_img = _read_image(body_side_path) if body_side_path else None
    face_bytes = Path(face_path).stat().st_size if Path(face_path).exists() else -1
    face_suffix = Path(face_path).suffix.lower()
    face_mime = mimetypes.guess_type(str(face_path))[0] or "unknown"
    body_front_bytes = Path(body_front_path).stat().st_size if Path(body_front_path).exists() else -1
    body_front_suffix = Path(body_front_path).suffix.lower()
    body_front_mime = mimetypes.guess_type(str(body_front_path))[0] or "unknown"
    logger.info(
        "face input metadata request_id=%s face_path=%s ext=%s mime=%s bytes=%s",
        request_id,
        face_path,
        face_suffix,
        face_mime,
        face_bytes,
    )
    logger.info(
        "pose input metadata request_id=%s body_front_path=%s ext=%s mime=%s bytes=%s",
        request_id,
        body_front_path,
        body_front_suffix,
        body_front_mime,
        body_front_bytes,
    )
    if face_img is None:
        logger.error(
            "face decode status request_id=%s decode_success=%s face_path=%s bytes=%s",
            request_id,
            False,
            face_path,
            face_bytes,
        )
        heic_hint = False
        if face_raw:
            heic_hint = (b"ftypheic" in face_raw[:64]) or (b"ftypheif" in face_raw[:64])
        raise AppError(
            error_code="INPUT_INVALID",
            message="Kunne ikke lese ansiktsbildet.",
            status_code=400,
            details={
                "face_path": face_path,
                "request_id": request_id,
                "mime": face_mime,
                "bytes": face_bytes,
                "decode_success": False,
                "format_hint": "heic/heif" if heic_hint else None,
                "hint": "Hvis bildet er HEIC (iPhone), konverter til JPG/PNG før opplasting." if heic_hint else None,
            },
        )
    if body_img is None:
        logger.error(
            "pose decode status request_id=%s decode_success=%s body_front_path=%s bytes=%s",
            request_id,
            False,
            body_front_path,
            body_front_bytes,
        )
        raise AppError(
            error_code="POSE_IMAGE_DECODE_FAILED",
            message="Kunne ikke dekode bodyFront-bildet.",
            status_code=400,
            details={
                "body_front_path": body_front_path,
                "request_id": request_id,
                "mime": body_front_mime,
                "bytes": body_front_bytes,
                "decode_success": False,
            },
        )
    logger.info(
        "pose decode status request_id=%s decode_success=%s image_shape=%s dtype=%s channels=%s",
        request_id,
        True,
        tuple(body_img.shape),
        str(body_img.dtype),
        int(body_img.shape[2]) if body_img.ndim == 3 else 1,
    )
    h_body, w_body = body_img.shape[:2]
    if w_body < MIN_BODY_FRONT_WIDTH_PX or h_body < MIN_BODY_FRONT_HEIGHT_PX:
        raise AppError(
            error_code="INPUT_INVALID",
            message="bodyFront er for lite. Last opp et tydelig full-body bilde.",
            status_code=400,
            details={
                "request_id": request_id,
                "min_width_px": MIN_BODY_FRONT_WIDTH_PX,
                "min_height_px": MIN_BODY_FRONT_HEIGHT_PX,
                "actual_width_px": int(w_body),
                "actual_height_px": int(h_body),
            },
        )
    if not ENABLE_BODY_ANALYSIS:
        raise AppError(
            error_code="FEATURE_DISABLED",
            message="Kroppsanalyse er deaktivert på serveren.",
            status_code=503,
            details={"request_id": request_id, "feature": "ENABLE_BODY_ANALYSIS"},
            retryable=False,
        )
    logger.info(
        "bodyFront image stats: shape=%s dtype=%s mode=%s min_px=%s max_px=%s",
        tuple(body_img.shape),
        body_img.dtype,
        "RGBA" if (body_img.ndim == 3 and body_img.shape[2] == 4) else "RGB" if (body_img.ndim == 3 and body_img.shape[2] == 3) else f"OTHER({body_img.shape[2] if body_img.ndim == 3 else 1})",
        int(np.min(body_img)),
        int(np.max(body_img)),
    )

    warnings = []
    prog(35, "Estimating body proportions")
    if body_scales is not None:
        resolved_scales = dict(body_scales)
        measurements = {}
        references = {}
        formulas = {}
        pose_debug = {"keypoints_count": None, "pose_confidence": None, "person_bbox_px": None}
    else:
        try:
            estimated_scales, measurements, references, formulas, pose_debug = _estimate_body_scales(
                body_img,
                body_side_img,
                request_id=request_id,
            )
        except AppError:
            raise
        except Exception as e:
            classified = classify_pose_exception(e)
            details = dict(classified.details or {})
            details.update(
                {
                    "request_id": request_id,
                    "body_front_path": body_front_path,
                    "body_front_bytes": body_front_bytes,
                    "body_front_shape": tuple(body_img.shape),
                }
            )
            classified.details = details
            raise classified from e
        resolved_scales = estimated_scales
    logger.info(
        "body analysis result: request_id=%s keypoints_count=%s pose_confidence=%s bodyFront_shape=%s",
        request_id,
        pose_debug.get("keypoints_count"),
        pose_debug.get("pose_confidence"),
        tuple(body_img.shape),
    )
    logger.info("body scales: %s", resolved_scales)
    if all(abs(float(resolved_scales.get(k, 1.0)) - 1.0) < 1e-6 for k in BODY_SCALE_DEFAULTS.keys()):
        msg = "all body scales are 1.0 (no effective personalization detected)"
        logger.warning(msg)
        warnings.append("no_effective_personalization")
    if measurements:
        logger.info("body measurements(px): %s", measurements)
        logger.info("body references(px): %s", references)
        logger.info("body formulas: %s", formulas)
        h_ratio = measurements.get("totalHeightRatio")
        if h_ratio is not None and float(h_ratio) < 0.45:
            msg = f"bodyFront er ikke full-body nok (totalHeightRatio={h_ratio})"
            logger.error("INPUT_INVALID: %s", msg)
            raise AppError(
                error_code="INPUT_INVALID",
                message=msg,
                status_code=400,
                details={"request_id": request_id, "totalHeightRatio": float(h_ratio)},
                retryable=False,
            )

    if EXTREME_DEBUG_DEFORM:
        logger.warning("EXTREME_DEBUG_DEFORM active: ignoring computed body scales for export deformation test")
        resolved_scales = dict(BODY_SCALE_DEFAULTS)

    prog(25, "Reading face metrics")
    face_rgb = face_img if (face_img.ndim == 3 and face_img.shape[-1] == 3) else cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_region = get_face_region(face_rgb)
    face_landmark_points, face_landmark_source = _get_face_landmarks_points(face_rgb)
    face_analysis = analyze_face_landmarks(face_rgb, face_landmark_points) if face_landmark_points is not None else {
        "faceParams": default_face_params(),
        "faceMetrics": default_face_metrics(),
        "avatarMorphParameters": apply_face_morphs(default_face_metrics(), avatar_style=avatar_style),
        "faceColorRgb8": None,
        "faceLandmarksCount": 0,
        "faceAnalysisSource": face_landmark_source if face_landmark_source in {"fallback", "failed"} else "failed",
    }
    if face_landmark_source == "fallback" and face_analysis.get("faceAnalysisSource") == "failed":
        face_analysis["faceAnalysisSource"] = "fallback"
    face_detection_confidence = 1.0 if face_region else 0.0

    prog(45, "Loading human base model")
    base_model_path = _get_base_mannequin_path(avatar_style)
    base_morph_debug = gltf_debug_morphs(str(base_model_path)) if base_model_path else {
        "ok": False,
        "hasMorphTargets": False,
        "meshCount": 0,
        "meshesWithMorphTargets": 0,
        "totalMorphTargets": 0,
        "meshMorphCounts": {},
        "meshMorphNames": {},
        "hasUnderwearMesh": False,
        "underwearMeshes": [],
        "error": "base_model_missing",
    }
    allow_vertex_deform = bool(base_morph_debug.get("hasMorphTargets"))
    if not allow_vertex_deform:
        warnings.append("morphs_missing_deformation_skipped")
    if not bool(base_morph_debug.get("hasUnderwearMesh")):
        warnings.append("underwear_mesh_missing_in_base")
    logger.info(
        "base mannequin morph debug request_id=%s style=%s hasMorphTargets=%s meshesWithMorphTargets=%s totalMorphTargets=%s hasUnderwearMesh=%s",
        request_id,
        avatar_style or "neutral",
        bool(base_morph_debug.get("hasMorphTargets")),
        int(base_morph_debug.get("meshesWithMorphTargets") or 0),
        int(base_morph_debug.get("totalMorphTargets") or 0),
        bool(base_morph_debug.get("hasUnderwearMesh")),
    )
    scene = _load_base_mannequin_scene(
        resolved_scales,
        avatar_style=avatar_style,
        allow_vertex_deform=allow_vertex_deform,
    )
    if scene is None:
        raise AppError(
            error_code="BASE_MODEL_UNAVAILABLE",
            message="Kunne ikke laste base-modell for avatar. Kontroller at base_human_male.glb/base_human_female.glb finnes og er gyldig.",
            status_code=500,
            details={
                "request_id": request_id,
                "avatarStyle": avatar_style,
                "baseDir": str(BASE_MANNEQUIN_DIR.resolve()),
                "baseModelPath": str(base_model_path.resolve()) if base_model_path else None,
            },
            retryable=True,
        )
    base_model_used = (getattr(scene, "metadata", {}) or {}).get("basePathUsed")
    if not base_model_used:
        base_model_used = str(base_model_path.resolve()) if base_model_path else None
    morph_inspect = inspect_head_morph_targets(str(base_model_path)) if base_model_path else {
        "hasHeadMorphTargets": False,
        "morphTargetNamesByMesh": {},
        "headMeshes": [],
        "morphTargetsFoundCount": 0,
        "method": "morph_unavailable",
    }
    logger.info(
        "face morph inspect request_id=%s hasHeadMorphTargets=%s morphTargetsFoundCount=%s headMeshes=%s morphTargetNamesByMesh=%s",
        request_id,
        bool(morph_inspect.get("hasHeadMorphTargets")),
        int(morph_inspect.get("morphTargetsFoundCount") or 0),
        morph_inspect.get("headMeshes"),
        morph_inspect.get("morphTargetNamesByMesh"),
    )
    avatar_bbox_before = _scene_bbox(scene)
    face_likeness_debug = {
        "applied": False,
        "method": "none",
        "headTargetsMatched": [],
        "scalingApplied": {},
        "morphTargetsApplied": [],
        "morphTargetNamesByMesh": dict(morph_inspect.get("morphTargetNamesByMesh") or {}),
        "hasHeadMorphTargets": bool(morph_inspect.get("hasHeadMorphTargets")),
        "morphTargetsFoundCount": int(morph_inspect.get("morphTargetsFoundCount") or 0),
        "overlayApplied": False,
        "overlayOpacity": 0.35,
        "beardDetected": False,
        "beardOpacity": 0.0,
        "textureApplied": False,
        "textureResolution": None,
        "headMaterialName": None,
        "morphWeights": {},
        "morphParameters": {},
        "likenessScore": 0.0,
        "likenessThreshold": 0.86,
        "likenessIterations": 0,
        "likenessValidationMethod": "not_run",
        "faceBoundingBoxPx": None,
        "alignmentApplied": False,
    }
    face_analysis_source = str(face_analysis.get("faceAnalysisSource", "failed"))
    face_params = dict(face_analysis.get("faceParams") or default_face_params())
    face_metrics = dict(face_analysis.get("faceMetrics") or default_face_metrics())
    morph_parameters = dict(face_analysis.get("avatarMorphParameters") or apply_face_morphs(face_metrics, avatar_style=avatar_style))
    likeness_threshold = 0.90 if str(avatar_style or "").strip().lower() == "female" else 0.86
    likeness_iters = 6 if str(avatar_style or "").strip().lower() == "female" else 4
    likeness_validation = validate_likeness_and_refine(
        face_metrics,
        morph_parameters,
        face_rgb=face_rgb,
        landmarks_xy_px=face_landmark_points,
        avatar_style=avatar_style,
        threshold=likeness_threshold,
        max_iters=likeness_iters,
    )
    morph_parameters = dict(likeness_validation.get("bestMorphParameters") or morph_parameters)
    face_likeness_debug["morphParameters"] = dict(morph_parameters)
    face_likeness_debug["likenessScore"] = float(likeness_validation.get("likenessScore") or 0.0)
    face_likeness_debug["likenessScoreMetric"] = float(likeness_validation.get("likenessScoreMetric") or 0.0)
    face_likeness_debug["likenessScoreEmbedding"] = float(likeness_validation.get("likenessScoreEmbedding") or 0.0)
    face_likeness_debug["likenessThreshold"] = float(likeness_validation.get("threshold") or likeness_threshold)
    face_likeness_debug["likenessIterations"] = int(likeness_validation.get("iterations") or 0)
    face_likeness_debug["likenessValidationMethod"] = str(likeness_validation.get("validationMethod") or "proxy_front_embedding")
    face_likeness_debug["embeddingEnabled"] = bool(likeness_validation.get("embeddingEnabled"))
    if face_analysis_source == "facemesh":
        if not bool(morph_inspect.get("hasHeadMorphTargets")):
            # No morph targets in base GLB -> keep current safe scale fallback and add texture overlay fallback.
            try:
                face_likeness_debug.update(apply_face_likeness_to_scene(scene, face_params, morph_parameters))
            except Exception as e:
                logger.warning("face likeness scale fallback failed request_id=%s err=%s", request_id, e)
                face_likeness_debug.update(
                    {
                        "applied": False,
                        "method": "scale_failed",
                        "headTargetsMatched": [],
                        "scalingApplied": {},
                        "morphTargetsApplied": [],
                    }
                )
            try:
                tex_overlay = apply_texture_overlay_fallback(
                    scene,
                    face_rgb,
                    face_landmark_points,
                    request_id=request_id,
                )
                if tex_overlay.get("applied"):
                    face_likeness_debug["method"] = "texture_overlay"
                    face_likeness_debug["applied"] = True
                    face_likeness_debug["textureOverlayMaterials"] = tex_overlay.get("materials", [])
                face_likeness_debug["overlayApplied"] = bool(tex_overlay.get("applied"))
                face_likeness_debug["textureApplied"] = bool(tex_overlay.get("applied"))
                face_likeness_debug["overlayOpacity"] = float(tex_overlay.get("overlayOpacity") or 0.35)
                face_likeness_debug["beardDetected"] = bool(tex_overlay.get("beardDetected"))
                face_likeness_debug["beardOpacity"] = float(tex_overlay.get("beardOpacity") or 0.0)
                face_likeness_debug["textureResolution"] = tex_overlay.get("textureResolution")
                face_likeness_debug["headMaterialName"] = tex_overlay.get("headMaterialName")
                face_likeness_debug["faceBoundingBoxPx"] = tex_overlay.get("faceBoundingBoxPx")
                face_likeness_debug["alignmentApplied"] = bool(tex_overlay.get("alignmentApplied"))
                face_likeness_debug["textureMaterialApplyFailures"] = int(tex_overlay.get("materialApplyFailures") or 0)
            except Exception as e:
                logger.warning("face texture overlay fallback failed request_id=%s err=%s", request_id, e)
                face_likeness_debug["textureOverlayMaterials"] = []
        else:
            face_likeness_debug["method"] = "none"
            face_likeness_debug["headTargetsMatched"] = list((morph_inspect.get("morphTargetNamesByMesh") or {}).keys())
            # Keep subtle overlay ON also when morphs exist.
            try:
                tex_overlay = apply_texture_overlay_fallback(
                    scene,
                    face_rgb,
                    face_landmark_points,
                    request_id=request_id,
                )
                face_likeness_debug["overlayApplied"] = bool(tex_overlay.get("applied"))
                face_likeness_debug["textureApplied"] = bool(tex_overlay.get("applied"))
                face_likeness_debug["overlayOpacity"] = float(tex_overlay.get("overlayOpacity") or 0.35)
                face_likeness_debug["beardDetected"] = bool(tex_overlay.get("beardDetected"))
                face_likeness_debug["beardOpacity"] = float(tex_overlay.get("beardOpacity") or 0.0)
                face_likeness_debug["textureResolution"] = tex_overlay.get("textureResolution")
                face_likeness_debug["headMaterialName"] = tex_overlay.get("headMaterialName")
                face_likeness_debug["faceBoundingBoxPx"] = tex_overlay.get("faceBoundingBoxPx")
                face_likeness_debug["alignmentApplied"] = bool(tex_overlay.get("alignmentApplied"))
                face_likeness_debug["textureMaterialApplyFailures"] = int(tex_overlay.get("materialApplyFailures") or 0)
            except Exception:
                pass
    else:
        warnings.append("face_analysis_failed")
    export_norm_debug = dict((getattr(scene, "metadata", {}) or {}).get("exportNormalization", {}) or {})
    root_transform_before = dict((getattr(scene, "metadata", {}) or {}).get("rootTransformBefore", {}) or {})
    root_transform_after = dict((getattr(scene, "metadata", {}) or {}).get("rootTransformAfter", {}) or {})

    # D) Sanitize materials FIRST so nothing overwrites skin later.
    prog(60, "Validating materials")
    _log_and_sanitize_scene_materials(scene)
    _log_body_node_transforms(scene)

    # Human mode with image-driven skin sampling.
    skin_srgb = None
    skin_debug = None
    materials_changed = []
    materials_debug_before_after = []
    skin_material_not_found = False
    skin_applied_visible = False
    skin_texture_removed = False
    male_hair_forced_black = []
    try:
        # 1) Sample skin from face image first.
        face_srgb, face_debug = estimate_skin_color_from_face(face_path)
        job_id = _infer_job_id_from_upload_path(face_path) or _infer_job_id_from_upload_path(body_front_path)
        seed_rgb8 = face_debug.get("rgb8") if isinstance(face_debug, dict) else None
        body_srgb, body_debug = estimate_skin_color_from_body_front(
            body_img,
            seed_rgb8=seed_rgb8,
            person_bbox_px=pose_debug.get("person_bbox_px"),
            job_id=job_id,
        )

        skin_srgb = face_srgb
        skin_debug = dict(face_debug or {})
        skin_debug["body"] = body_debug
        face_method = str((face_debug or {}).get("method", ""))
        face_px = int((face_debug or {}).get("pixels_used", 0)) if isinstance(face_debug, dict) else 0
        body_px = int((body_debug or {}).get("pixels_used", 0)) if isinstance(body_debug, dict) else 0
        face_valid = bool((not face_method.startswith("default_")) and face_px > 0)
        body_valid = bool(body_srgb is not None and body_px >= 250)

        if face_valid:
            skin_debug["source"] = "face_only"
            # Optional mild refine from body when available.
            if body_valid:
                body_weight = min(0.20, float(body_px) / float(max(1, face_px + body_px)))
                face_weight = 1.0 - body_weight
                skin_srgb = (
                    float(face_srgb[0]) * face_weight + float(body_srgb[0]) * body_weight,
                    float(face_srgb[1]) * face_weight + float(body_srgb[1]) * body_weight,
                    float(face_srgb[2]) * face_weight + float(body_srgb[2]) * body_weight,
                )
                skin_debug["source"] = "face_primary+body_refine"
                skin_debug["blend_weights"] = {"face": face_weight, "body": body_weight}
                skin_debug["rgb8"] = [
                    int(round(float(skin_srgb[0]) * 255)),
                    int(round(float(skin_srgb[1]) * 255)),
                    int(round(float(skin_srgb[2]) * 255)),
                ]
                skin_debug["srgb_factor"] = [float(skin_srgb[0]), float(skin_srgb[1]), float(skin_srgb[2])]
        else:
            # 3) Do not silently fallback; log reason and expose debug source.
            logger.warning(
                "face skin sampling failed/fallback request_id=%s method=%s face_px=%s face_conf=%s face_path=%s",
                request_id,
                face_method,
                face_px,
                face_detection_confidence,
                str(Path(face_path).resolve()),
            )
            if body_valid:
                skin_srgb = body_srgb
                skin_debug["source"] = "body_fallback"
                skin_debug["rgb8"] = [
                    int(round(float(skin_srgb[0]) * 255)),
                    int(round(float(skin_srgb[1]) * 255)),
                    int(round(float(skin_srgb[2]) * 255)),
                ]
                skin_debug["srgb_factor"] = [float(skin_srgb[0]), float(skin_srgb[1]), float(skin_srgb[2])]
            else:
                skin_srgb = FIXED_HUMAN_SKIN_SRGB
                skin_debug["source"] = "fallback"
                skin_debug["fallbackReason"] = {
                    "faceMethod": face_method,
                    "facePixels": face_px,
                    "bodyPixels": body_px,
                    "faceDetectionConfidence": face_detection_confidence,
                }
                skin_debug["rgb8"] = [
                    int(round(float(skin_srgb[0]) * 255)),
                    int(round(float(skin_srgb[1]) * 255)),
                    int(round(float(skin_srgb[2]) * 255)),
                ]
                skin_debug["srgb_factor"] = [float(skin_srgb[0]), float(skin_srgb[1]), float(skin_srgb[2])]

        forced_debug_rgb = None
        if DEBUG_FORCE_SKIN_COLOR:
            try:
                forced = [int(x.strip()) for x in DEBUG_FORCE_SKIN_COLOR.split(",")]
                if len(forced) == 3:
                    forced_debug_rgb = [int(np.clip(v, 0, 255)) for v in forced]
                    skin_srgb = (
                        float(forced_debug_rgb[0]) / 255.0,
                        float(forced_debug_rgb[1]) / 255.0,
                        float(forced_debug_rgb[2]) / 255.0,
                    )
                    skin_debug["source"] = "debug_force_skin_color"
                    skin_debug["rgb8"] = list(forced_debug_rgb)
                    skin_debug["srgb_factor"] = [float(skin_srgb[0]), float(skin_srgb[1]), float(skin_srgb[2])]
            except Exception:
                logger.warning("invalid DEBUG_FORCE_SKIN_COLOR value=%s", DEBUG_FORCE_SKIN_COLOR)

        # 4/5) Ensure sampled skin is applied before export.
        materials_changed, materials_debug_before_after, skin_material_not_found, skin_applied_visible, skin_texture_removed = apply_skin_color_to_scene(scene, skin_srgb)
        male_hair_forced_black = enforce_male_hair_black(scene, avatar_style)
        if male_hair_forced_black:
            logger.info("male hair enforced black request_id=%s materials=%s", request_id, male_hair_forced_black)
        pipeline_material_changes = []
        for row in materials_debug_before_after:
            logger.info(
                "skin material debug: name=%s mesh=%s hasBaseColorTexture=%s hasBaseColorTextureAfter=%s factorSet=%s before=%s after=%s",
                row.get("materialName"),
                row.get("meshNames"),
                row.get("hasBaseColorTexture"),
                row.get("hasBaseColorTextureAfter"),
                row.get("baseColorFactorSet"),
                row.get("baseColorFactorBefore"),
                row.get("baseColorFactorAfter"),
            )
            if row.get("applied"):
                pipeline_material_changes.append(
                    {
                        "materialName": row.get("materialName"),
                        "meshNames": row.get("meshNames"),
                        "baseColorTextureSet": bool(row.get("hasBaseColorTextureAfter")),
                        "baseColorFactorSet": bool(row.get("baseColorFactorSet")),
                    }
                )
        material_summary = _material_debug_summary(scene, top_n=8)
        logger.info(
            "skin color applied: request_id=%s srgb=%s rgb8=%s materials=%s method=%s face_px=%s",
            request_id,
            [round(float(x), 4) for x in skin_srgb],
            skin_debug.get("rgb8") if isinstance(skin_debug, dict) else None,
            materials_changed,
            skin_debug.get("method") if isinstance(skin_debug, dict) else None,
            skin_debug.get("pixels_used") if isinstance(skin_debug, dict) else None,
        )
    except Exception as e:
        logger.warning("skin color apply failed (alt1) request_id=%s err=%s", request_id, e)
        material_summary = []
        pipeline_material_changes = []
        forced_debug_rgb = None
        skin_material_not_found = True
        skin_applied_visible = False
        skin_texture_removed = False

    used_debug_deform = False
    if DEBUG_VISUAL_DEFORM:
        used_debug_deform = True
        avatar_bbox_before, avatar_bbox_after, deform_warnings = _apply_debug_visual_deform(scene, resolved_scales)
        warnings.extend(deform_warnings)
    else:
        avatar_bbox_after = _scene_bbox(scene)

    prog(70, "Finalizing model")
    extreme_vertex_count = None
    if EXTREME_DEBUG_DEFORM:
        avatar_bbox_before, avatar_bbox_after, extreme_vertex_count = _apply_extreme_debug_deform(scene)
        warnings.append("extreme_debug_deform_active")
    out_path = Path(out_glb_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prog(85, "Exporting GLB")
    logger.info(
        "export normalization debug: request_id=%s rootBefore=%s rootAfter=%s bboxBefore=%s bboxAfter=%s axisFix=%s scaleFix=%s scaleFactor=%s",
        request_id,
        root_transform_before,
        root_transform_after,
        export_norm_debug.get("bboxBefore"),
        export_norm_debug.get("bboxAfter"),
        export_norm_debug.get("axisFixApplied"),
        export_norm_debug.get("scaleFixApplied"),
        export_norm_debug.get("scaleFactor"),
    )
    try:
        scene.export(str(out_path), file_type="glb")
    except Exception as ex:
        logger.exception("GLB export failed: %s", ex)
        raise AppError(
            error_code="INTERNAL_ERROR",
            message="Kunne ikke eksportere GLB.",
            status_code=500,
            details={"request_id": request_id, "error": str(ex)},
            retryable=True,
        ) from ex

    # Human mode: do not run legacy skin postprocess (it may strip baseColor textures).
    skin_postprocess = {"applied": False, "method": "skipped_human_mode", "materials": []}

    # Optional morph pass on exported GLB (if morph targets exist).
    morph_debug = {
        "applied": False,
        "method": "morph_unavailable",
        "morphTargetsApplied": [],
        "morphTargetNamesByMesh": dict(morph_inspect.get("morphTargetNamesByMesh") or {}),
        "hasHeadMorphTargets": bool(morph_inspect.get("hasHeadMorphTargets")),
        "morphWeights": {},
    }
    if (
        str(face_analysis.get("faceAnalysisSource", "failed")) == "facemesh"
        and bool(morph_inspect.get("hasHeadMorphTargets"))
    ):
        try:
            morph_debug = apply_face_likeness_to_glb_morphs(str(out_path), face_params, morph_parameters)
        except Exception as e:
            logger.warning("face morph pass failed request_id=%s err=%s", request_id, e)
            morph_debug = {
                "applied": False,
                "method": "morph_failed",
                "morphTargetsApplied": [],
                "morphTargetNamesByMesh": dict(morph_inspect.get("morphTargetNamesByMesh") or {}),
                "hasHeadMorphTargets": bool(morph_inspect.get("hasHeadMorphTargets")),
                "morphWeights": {},
            }
    elif str(face_analysis.get("faceAnalysisSource", "failed")) == "facemesh":
        warnings.append("face_morphs_missing_fallback_used")
    if isinstance(face_likeness_debug, dict):
        face_likeness_debug["morphTargetsApplied"] = morph_debug.get("morphTargetsApplied", [])
        face_likeness_debug["morphTargetNamesByMesh"] = morph_debug.get("morphTargetNamesByMesh", {})
        face_likeness_debug["hasHeadMorphTargets"] = bool(morph_debug.get("hasHeadMorphTargets"))
        face_likeness_debug["morphTargetsFoundCount"] = sum(
            len(v) for v in (morph_debug.get("morphTargetNamesByMesh") or {}).values()
        )
        face_likeness_debug["morphWeights"] = dict(morph_debug.get("morphWeights") or {})
        if morph_debug.get("applied"):
            face_likeness_debug["method"] = "morph"

    # Logging: last tilbake og rapporter
    glb_inspection = {
        "meshInspection": [],
        "headMeshNames": [],
        "hairMeshNames": [],
        "eyeMeshNames": [],
        "morphTargetsByMesh": {},
        "hasHeadMorphTargets": False,
    }
    output_morph_debug = {
        "ok": False,
        "path": str(out_path.resolve()),
        "exists": False,
        "hasMorphTargets": False,
        "meshCount": 0,
        "meshesWithMorphTargets": 0,
        "totalMorphTargets": 0,
        "meshMorphCounts": {},
        "meshMorphNames": {},
        "nodeWeightsPresent": False,
        "meshWeightsPresent": False,
        "hasUnderwearMesh": False,
        "underwearMeshes": [],
        "error": "not_inspected",
    }
    try:
        glb_inspection = inspect_exported_glb_structure(str(out_path))
        output_morph_debug = gltf_debug_morphs(str(out_path))
        logger.info(
            "exported glb inspection request_id=%s head=%s hair=%s eyes=%s hasHeadMorphTargets=%s",
            request_id,
            glb_inspection.get("headMeshNames"),
            glb_inspection.get("hairMeshNames"),
            glb_inspection.get("eyeMeshNames"),
            glb_inspection.get("hasHeadMorphTargets"),
        )
        loaded = trimesh.load(out_glb_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
        else:
            geoms = [loaded]
        total_verts = sum(len(g.vertices) for g in geoms)
        total_faces = sum(len(g.faces) for g in geoms)
        logger.info("avatar glb: vertices=%s, faces=%s", total_verts, total_faces)
        all_bounds = np.vstack([np.asarray(g.bounds, dtype=float).reshape(2, 3) for g in geoms])
        bbox_min = all_bounds.min(axis=0)
        bbox_max = all_bounds.max(axis=0)
        size = np.asarray(bbox_max - bbox_min, dtype=float).reshape(-1)
        logger.info("avatar bbox size (x,y,z): %s", size.tolist())
        has_tex = any(getattr(g.visual, "material", None) and getattr(g.visual.material, "image", None) for g in geoms)
        logger.info("avatar has_textures=%s", has_tex)
        if isinstance(loaded, trimesh.Scene):
            _log_exported_scene_materials(loaded)
            exported_summary = _exported_materials_summary(loaded, top_n=8)
        else:
            tmp_scene = trimesh.Scene()
            tmp_scene.add_geometry(geoms)
            _log_exported_scene_materials(tmp_scene)
            exported_summary = _exported_materials_summary(tmp_scene, top_n=8)
    except Exception as e:
        logger.warning("Could not validate exported GLB: %s", e)
        total_verts = 0
        total_faces = 0
        has_tex = False
        exported_summary = []
        output_morph_debug = gltf_debug_morphs(str(out_path))
        skin_postprocess = skin_postprocess if "skin_postprocess" in locals() else None

    # E) Output hash to verify different inputs produce different outputs.
    output_file_hash = None
    try:
        out_bytes = out_path.read_bytes()
        output_file_hash = hashlib.sha256(out_bytes).hexdigest()
        logger.info("outputFileHash (E) request_id=%s hash=%s size=%s", request_id, output_file_hash[:16], len(out_bytes))
    except Exception:
        pass

    prog(100, "Done")

    # Material validation list from final scene.
    material_list = []
    has_normal_map = False
    try:
        if isinstance(loaded, trimesh.Scene):
            material_list = _material_list_for_debug(loaded)
        else:
            tmp_scene2 = trimesh.Scene()
            tmp_scene2.add_geometry(geoms)
            material_list = _material_list_for_debug(tmp_scene2)
        has_normal_map = any(bool(m.get("hasNormalMap")) for m in material_list)
    except Exception:
        material_list = []
        has_normal_map = False

    # A) sampledSkinRGB_srgb [0-255] and [0-1], sampledSkinLinear
    rgb8 = skin_debug.get("rgb8") if isinstance(skin_debug, dict) else None
    srgb_01 = [float(skin_srgb[0]), float(skin_srgb[1]), float(skin_srgb[2])] if skin_srgb else None
    linear = [float(_srgb_to_linear_channel(float(skin_srgb[0]))), float(_srgb_to_linear_channel(float(skin_srgb[1]))), float(_srgb_to_linear_channel(float(skin_srgb[2])))] if skin_srgb else None
    applied_base_color_factor = []
    try:
        for m in material_list:
            mn = str(m.get("materialName", "")).lower()
            if any(k in mn for k in SKIN_MATERIAL_KEYWORDS):
                applied_base_color_factor.append(
                    {
                        "materialName": m.get("materialName"),
                        "baseColorFactor": m.get("baseColorFactor"),
                        "hasBaseColorTexture": bool(m.get("hasBaseColorTexture")),
                    }
                )
    except Exception:
        applied_base_color_factor = []
    materials_skipped = []
    try:
        for row in materials_debug_before_after:
            if bool(row.get("skippedByRule")):
                materials_skipped.append(str(row.get("materialName")))
    except Exception:
        materials_skipped = []

    return {
        "avatarStyle": (avatar_style or "neutral"),
        "faceLikenessEnabled": bool(face_analysis_source == "facemesh"),
        "faceLikenessMethod": face_likeness_debug.get("method"),
        "faceAnalysisSource": face_analysis_source,
        "faceLandmarksCount": int(face_analysis.get("faceLandmarksCount") or 0),
        "faceParams": face_params,
        "faceMetrics": face_metrics,
        "morphParameters": dict(face_likeness_debug.get("morphParameters") or {}),
        "likenessScore": float(face_likeness_debug.get("likenessScore") or 0.0),
        "faceSimilarityScore": float(face_likeness_debug.get("likenessScore") or 0.0),
        "likenessScoreMetric": float(face_likeness_debug.get("likenessScoreMetric") or 0.0),
        "likenessScoreEmbedding": float(face_likeness_debug.get("likenessScoreEmbedding") or 0.0),
        "embeddingEnabled": bool(face_likeness_debug.get("embeddingEnabled")),
        "likenessValidationMethod": face_likeness_debug.get("likenessValidationMethod"),
        "faceMetricsDebug": {
            "eyeSpacing": float((face_likeness_debug.get("morphParameters") or {}).get("eyeSpacing", 0.0)),
            "noseWidth": float(face_metrics.get("noseWidth") or 0.0),
            "jawWidth": float(face_metrics.get("jawWidth") or 0.0),
            "mouthWidth": float(face_metrics.get("mouthWidth") or 0.0),
            "chinHeight": float((face_likeness_debug.get("morphParameters") or {}).get("chinHeight", 0.0)),
        },
        "morphParametersDebug": {
            "eyeSpacing": float((face_likeness_debug.get("morphParameters") or {}).get("eyeSpacing", 0.0)),
            "noseWidth": float((face_likeness_debug.get("morphParameters") or {}).get("noseWidth", 0.0)),
            "jawWidth": float((face_likeness_debug.get("morphParameters") or {}).get("jawWidth", 0.0)),
            "chinHeight": float((face_likeness_debug.get("morphParameters") or {}).get("chinHeight", 0.0)),
        },
        "headTargetsMatched": list(face_likeness_debug.get("headTargetsMatched") or []),
        "morphTargetsApplied": list(face_likeness_debug.get("morphTargetsApplied") or []),
        "morphFound": bool(face_likeness_debug.get("hasHeadMorphTargets")),
        "morphNames": dict(face_likeness_debug.get("morphTargetNamesByMesh") or {}),
        "morphWeights": dict(face_likeness_debug.get("morphWeights") or {}),
        "hasHeadMorphTargets": bool(face_likeness_debug.get("hasHeadMorphTargets")),
        "morphTargetNamesByMesh": dict(face_likeness_debug.get("morphTargetNamesByMesh") or {}),
        "morphTargetsFoundCount": int(face_likeness_debug.get("morphTargetsFoundCount") or 0),
        "scalingApplied": dict(face_likeness_debug.get("scalingApplied") or {}),
        "textureOverlayMaterials": list(face_likeness_debug.get("textureOverlayMaterials") or []),
        "textureApplied": bool(face_likeness_debug.get("textureApplied")),
        "textureResolution": face_likeness_debug.get("textureResolution"),
        "textureMaterialApplyFailures": int(face_likeness_debug.get("textureMaterialApplyFailures") or 0),
        "headMaterialName": face_likeness_debug.get("headMaterialName"),
        "faceBoundingBoxPx": face_likeness_debug.get("faceBoundingBoxPx"),
        "alignmentApplied": bool(face_likeness_debug.get("alignmentApplied")),
        "overlayApplied": bool(face_likeness_debug.get("overlayApplied")),
        "overlayOpacity": float(face_likeness_debug.get("overlayOpacity") or 0.35),
        "beardDetected": bool(face_likeness_debug.get("beardDetected")),
        "beardOpacity": float(face_likeness_debug.get("beardOpacity") or 0.0),
        "headMeshNames": list(glb_inspection.get("headMeshNames") or []),
        "hairMeshNames": list(glb_inspection.get("hairMeshNames") or []),
        "eyeMeshNames": list(glb_inspection.get("eyeMeshNames") or []),
        "morphTargetsByMesh": dict(glb_inspection.get("morphTargetsByMesh") or {}),
        "meshInspection": list(glb_inspection.get("meshInspection") or []),
        "faceColorRgb8": face_analysis.get("faceColorRgb8"),
        "measurementsPx": measurements,
        "body_measurements_px": measurements,
        "pose_keypoints_count": pose_debug.get("keypoints_count"),
        "keypoints_count": pose_debug.get("keypoints_count"),
        "pose_confidence": pose_debug.get("pose_confidence"),
        "person_bbox_px": pose_debug.get("person_bbox_px"),
        "scales": resolved_scales,
        "warnings": sorted(set(warnings)),
        "deformationApplied": bool(allow_vertex_deform),
        "deformationMode": "vertex_proportion" if allow_vertex_deform else "skipped_morphs_missing",
        "usedDebugDeform": used_debug_deform,
        "baseModelUsed": base_model_used,
        "baseMorphDebug": base_morph_debug,
        "outputMorphDebug": output_morph_debug,
        "outputHasMorphTargets": bool(output_morph_debug.get("hasMorphTargets")),
        "outputMorphTargetMeshCount": int(output_morph_debug.get("meshesWithMorphTargets") or 0),
        "outputMorphTargetTotal": int(output_morph_debug.get("totalMorphTargets") or 0),
        "underwearMeshes": list(output_morph_debug.get("underwearMeshes") or []),
        "hasUnderwearMesh": bool(output_morph_debug.get("hasUnderwearMesh")),
        "avatar_bbox_before": avatar_bbox_before,
        "avatar_bbox_after": avatar_bbox_after,
        "rootTransformBefore": root_transform_before,
        "rootTransformAfter": root_transform_after,
        "bboxBefore": export_norm_debug.get("bboxBefore", avatar_bbox_before),
        "bboxAfter": export_norm_debug.get("bboxAfter", avatar_bbox_after),
        "orientationFixApplied": bool(export_norm_debug.get("axisFixApplied", False)),
        "scaleFixApplied": bool(export_norm_debug.get("scaleFixApplied", False)),
        "exportScaleFactor": float(export_norm_debug.get("scaleFactor", 1.0) or 1.0),
        "polyCount": int(total_faces),
        "vertex_count": int(extreme_vertex_count if extreme_vertex_count is not None else total_verts),
        "face_count": int(total_faces),
        "has_textures": bool(has_tex),
        "hasNormalMap": bool(has_normal_map),
        "lightingMode": LIGHTING_MODE,
        "materialList": material_list,
        # A) Prove skin extraction changes per request
        "inputImageHash": face_image_hash,
        "inputImageHashBodyFront": body_front_image_hash,
        "inputImageHashBodySide": body_side_image_hash,
        "faceFilePath": str(Path(face_path).resolve()),
        "bodyFrontFilePath": str(Path(body_front_path).resolve()),
        "faceFileSize": face_file_size,
        "bodyFrontFileSize": body_front_file_size,
        "skinColorSource": skin_debug.get("source") if isinstance(skin_debug, dict) else None,
        "sampledSkinRGB_srgb": rgb8,
        "sampledSkinRGB_srgb_01": srgb_01,
        "sampledSkinLinear": linear,
        "appliedSkinSRGB01": srgb_01,
        "appliedSkinLinear": linear,
        "appliedBaseColorFactor": applied_base_color_factor,
        "numberOfPixelsUsed": skin_debug.get("pixels_used") if isinstance(skin_debug, dict) else None,
        "facePixelsUsed": skin_debug.get("pixels_used") if isinstance(skin_debug, dict) else None,
        "bodyPixelsUsed": (skin_debug.get("body") or {}).get("pixels_used") if isinstance(skin_debug, dict) else None,
        "faceDetectionConfidence": face_detection_confidence,
        # C) Material debug (every material row with meshNames and before/after)
        "materialsSkinApplied": materials_debug_before_after,
        "materialDebug": materials_debug_before_after,
        "skinMaterialNotFound": bool(skin_material_not_found),
        "skinAppliedVisible": bool(skin_applied_visible),
        "skinTextureRemoved": bool(skin_texture_removed),
        "maleHairForcedBlackMaterials": list(male_hair_forced_black or []),
        "skinMaterialsChanged": materials_changed,
        "materialsSkipped": sorted(set(materials_skipped)),
        # Legacy / compatibility
        "skinColorRgb": rgb8,
        "skinColorFactor": (list(skin_srgb) + [1.0]) if skin_srgb else None,
        "skinColorFactorLinear": (linear + [1.0]) if linear else None,
        "skinColorMethod": skin_debug.get("method") if isinstance(skin_debug, dict) else None,
        "skinPixelsUsed": skin_debug.get("pixels_used") if isinstance(skin_debug, dict) else None,
        "skinBodyPixelsUsed": (skin_debug.get("body") or {}).get("pixels_used") if isinstance(skin_debug, dict) else None,
        "skinBodyMethod": (skin_debug.get("body") or {}).get("method") if isinstance(skin_debug, dict) else None,
        "materialsSummary": material_summary,
        "pipelineMaterialChanges": pipeline_material_changes,
        "exportedMaterialsSummary": exported_summary,
        "skinPostprocess": skin_postprocess,
        "debugForceSkinColor": forced_debug_rgb,
        "debugFaceCropPath": skin_debug.get("debug_face_crop_path") if isinstance(skin_debug, dict) else None,
        "debugSkinMaskPath": skin_debug.get("debug_skin_mask_path") if isinstance(skin_debug, dict) else None,
        "debugBodyCropPath": (skin_debug.get("body") or {}).get("debug_body_crop_path") if isinstance(skin_debug, dict) else None,
        "debugBodySkinMaskPath": (skin_debug.get("body") or {}).get("debug_body_skin_mask_path") if isinstance(skin_debug, dict) else None,
        # E) Output verification
        "outputFileHash": output_file_hash,
        "outHash": output_file_hash,
        "outputPath": str(out_path.resolve()),
        "references": references,
        "formulas": formulas,
    }
