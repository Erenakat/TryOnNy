"""
AI Avatar Backend – avatar fra fjes + kropp (job-basert, GLB).
Endpoints: POST /avatar/jobs, GET /avatar/jobs/:jobId
Bruk: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import io
import base64
import uuid
import threading
import logging
import hashlib
import traceback
import os
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np

from jobs import create_job, get_job, update_job, JobStatus
from pipeline import run_pipeline, get_face_region, ensure_models_loaded, _decode_image_bytes, debug_mannequin_morphs  # type: ignore
from errors import AppError, error_response_payload, is_dev_mode, classify_pose_exception, sanitize_for_json
from pose_service import pose_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Avatar AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
AVATARS_DIR = BASE_DIR / "static" / "avatars"
DEBUG_INPUTS_DIR = BASE_DIR / "static" / "debug_inputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

_recent_errors_by_request_id: dict[str, dict] = {}
_recent_uploads: deque[dict] = deque(maxlen=200)


def _remember_error(request_id: str, payload: dict) -> None:
    # Keep last ~200 errors in memory for dev debugging.
    try:
        payload = sanitize_for_json(payload)
        payload["ts"] = int(time.time())
        _recent_errors_by_request_id[request_id] = payload
        if len(_recent_errors_by_request_id) > 200:
            # drop oldest
            oldest = sorted(_recent_errors_by_request_id.items(), key=lambda kv: kv[1].get("ts", 0))[:50]
            for k, _ in oldest:
                _recent_errors_by_request_id.pop(k, None)
    except Exception:
        logger.exception("failed to remember error request_id=%s", request_id)


def _remember_upload(payload: dict) -> None:
    try:
        _recent_uploads.appendleft(sanitize_for_json(payload))
    except Exception:
        logger.exception("failed to remember upload payload")


def _get_request_id(request: Request | None = None) -> str:
    if request is not None:
        rid = getattr(request.state, "request_id", None)
        if rid:
            return rid
    return str(uuid.uuid4())


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get("x-request-id")
    request_id = incoming or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    request_id = _get_request_id(request)
    logger.exception(
        "app error request_id=%s code=%s status=%s details=%s",
        request_id,
        exc.error_code,
        exc.status_code,
        exc.details,
    )
    _remember_error(
        request_id,
        {
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response_payload(
            error_code=exc.error_code,
            message=exc.message,
            request_id=request_id,
            retryable=exc.retryable,
            details=sanitize_for_json(exc.details) if exc.details else None,
        ),
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    request_id = _get_request_id(request)
    logger.exception("http error request_id=%s status=%s detail=%s", request_id, exc.status_code, exc.detail)
    details = {"http_detail": sanitize_for_json(exc.detail)}
    _remember_error(
        request_id,
        {
            "error_code": "HTTP_EXCEPTION",
            "message": str(exc.detail),
            "details": details,
            "status_code": exc.status_code,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response_payload(
            error_code="INPUT_INVALID" if exc.status_code < 500 else "INTERNAL_ERROR",
            message=str(exc.detail),
            request_id=request_id,
            retryable=False,
            details=details,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = _get_request_id(request)
    logger.exception("unhandled error request_id=%s", request_id)
    details = {"exception": type(exc).__name__, "stacktrace": traceback.format_exc()}
    _remember_error(
        request_id,
        {
            "error_code": "INTERNAL_ERROR",
            "message": "Uventet serverfeil.",
            "details": details,
        },
    )
    return JSONResponse(
        status_code=500,
        content=error_response_payload(
            error_code="INTERNAL_ERROR",
            message="Uventet serverfeil.",
            request_id=request_id,
            retryable=False,
            details=details,
        ),
    )


@app.get("/static/avatars/{filename:path}")
def get_avatar_glb(filename: str):
    target = (AVATARS_DIR / filename).resolve()
    avatars_root = AVATARS_DIR.resolve()
    if not str(target).startswith(str(avatars_root)) or not target.is_file():
        raise HTTPException(404, "File not found")

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    return FileResponse(path=str(target), media_type="model/gltf-binary", headers=headers)


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.on_event("startup")
def _startup_load_models():
    face_ok = ensure_models_loaded()
    pose_ok = pose_service.init()
    logger.info(
        "startup detector init worker_pid=%s face_initialized=%s pose_initialized=%s",
        os.getpid(),
        face_ok,
        pose_ok,
    )


@app.get("/debug/pose-health")
def pose_health():
    try:
        health = pose_service.health()
        return health
    except Exception as e:
        logger.exception("pose health endpoint failed")
        return {
            "available": False,
            "error_code": "POSE_HEALTH_FAILED",
            "error_message": str(e),
            "deps": {"mediapipe": False, "cv2": False},
            "model": "mediapipe_pose",
        }


@app.get("/debug/mannequin-morphs")
def mannequin_morphs(avatarStyle: str = "neutral"):
    if not is_dev_mode():
        raise AppError(
            error_code="FEATURE_DISABLED",
            message="Debug endepunkt er deaktivert.",
            status_code=404,
            retryable=False,
        )
    style = (avatarStyle or "neutral").strip().lower()
    if style not in ("neutral", "male", "female"):
        style = "neutral"
    return debug_mannequin_morphs(style)


@app.get("/debug/last-error/{request_id}")
def debug_last_error(request_id: str):
    if not is_dev_mode():
        raise AppError(
            error_code="FEATURE_DISABLED",
            message="Debug endepunkt er deaktivert.",
            status_code=404,
            retryable=False,
        )
    payload = _recent_errors_by_request_id.get(request_id)
    if not payload:
        return {"found": False, "request_id": request_id}
    return {"found": True, "request_id": request_id, "error": payload}


@app.get("/debug/last-upload")
def debug_last_upload():
    if not is_dev_mode():
        raise AppError(
            error_code="FEATURE_DISABLED",
            message="Debug endepunkt er deaktivert.",
            status_code=404,
            retryable=False,
        )
    items = []
    for it in list(_recent_uploads)[:5]:
        items.append(
            {
                "jobId": it.get("jobId"),
                "requestId": it.get("requestId"),
                "hashes": it.get("hashes"),
                "debugInputPaths": it.get("debugInputPaths"),
            }
        )
    return {"count": len(items), "jobs": items}


def extract_face(image: np.ndarray) -> np.ndarray | None:
    """Hent fjes fra bilde med MediaPipe Tasks Face Landmarker."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    region = get_face_region(rgb)
    if not region:
        return None
    x1, y1, x2, y2 = region
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face


def make_avatar(face_img: np.ndarray) -> bytes:
    """Lag avatar: sirkulært fjes på kroppsfigur (PNG fallback)."""
    size = 400
    canvas = np.ones((size, 280, 3), dtype=np.uint8) * 255
    face_h, face_w = face_img.shape[:2]
    dim = min(face_h, face_w)
    face_crop = face_img[:dim, :dim] if face_h >= face_w else face_img[:, :dim]
    face_small = cv2.resize(face_crop, (140, 140))
    mask = np.zeros((140, 140), dtype=np.uint8)
    cv2.circle(mask, (70, 70), 68, 255, -1)
    y_offset, x_center = 30, 140
    roi = canvas[y_offset : y_offset + 140, x_center - 70 : x_center + 70]
    if roi.shape[:2] == (140, 140):
        roi[:] = cv2.bitwise_and(face_small, face_small, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        roi[inv_mask > 0] = 255
    body_color = (232, 228, 224)
    pts = np.array([[x_center - 50, 175], [x_center + 50, 175], [x_center + 45, 370], [x_center - 45, 370]], np.int32)
    cv2.fillPoly(canvas, [pts], body_color)
    cv2.ellipse(canvas, (x_center, 165), (55, 25), 0, 0, 360, body_color, -1)
    _, buf = cv2.imencode(".png", canvas)
    return buf.tobytes()


def _run_job(job_id: str, request_id: str) -> None:
    job = get_job(job_id)
    if not job or job.status != JobStatus.queued:
        return
    update_job(job_id, status=JobStatus.processing, progress=0, request_id=request_id)
    out_glb = AVATARS_DIR / f"{job_id}.glb"

    def progress_cb(pct: int, msg: str = "") -> None:
        update_job(job_id, progress=pct, progress_message=msg or None)
        logger.info("job %s progress %s %s", job_id, pct, msg)

    try:
        debug_payload = run_pipeline(
            str(job.face_path),
            str(job.body_front_path),
            str(job.body_side_path) if job.body_side_path else None,
            str(out_glb),
            progress_cb=progress_cb,
            request_id=request_id,
            avatar_style=job.avatar_style,
        )
        debug_payload = sanitize_for_json(debug_payload)
        out_bytes = out_glb.read_bytes()
        out_size = len(out_bytes)
        out_sha256 = hashlib.sha256(out_bytes).hexdigest()
        logger.info(
            "avatar output: job_id=%s output_path=%s size_bytes=%s sha256=%s",
            job_id,
            out_glb.resolve(),
            out_size,
            out_sha256,
        )
        avatar_url = f"/static/avatars/{job_id}.glb"
        update_job(job_id, status=JobStatus.done, progress=100, avatar_url=avatar_url, body_debug=debug_payload, request_id=request_id)
        logger.info("job %s done -> %s", job_id, avatar_url)
    except AppError as e:
        logger.exception("job %s failed request_id=%s code=%s", job_id, request_id, e.error_code)
        update_job(
            job_id,
            status=JobStatus.failed,
            error=e.message,
            error_code=e.error_code,
            error_details=sanitize_for_json(e.details) if (is_dev_mode() and e.details) else None,
            retryable=e.retryable,
            request_id=request_id,
        )
    except Exception as e:
        pose_error = classify_pose_exception(e)
        logger.exception("job %s failed request_id=%s code=%s", job_id, request_id, pose_error.error_code)
        details = {"exception": type(e).__name__, "raw_error": str(e)}
        if is_dev_mode():
            details["stacktrace"] = traceback.format_exc()
        update_job(
            job_id,
            status=JobStatus.failed,
            error=pose_error.message,
            error_code=pose_error.error_code,
            error_details=sanitize_for_json(details) if is_dev_mode() else None,
            retryable=pose_error.retryable,
            request_id=request_id,
        )


# --- Job API ---

class JobCreateResponse(BaseModel):
    jobId: str


class JobStatusResponse(BaseModel):
    status: str
    progress: int | None = None
    progress_message: str | None = None
    avatarUrl: str | None = None
    bodyDebug: dict | None = None
    error: str | None = None
    message: str | None = None
    error_code: str | None = None
    request_id: str | None = None
    retryable: bool | None = None
    details: dict | None = None
    avatarStyle: str | None = None


@app.post("/avatar/jobs", response_model=JobCreateResponse)
async def create_avatar_job(
    request: Request,
    face: UploadFile = File(...),
    bodyFront: UploadFile = File(...),
    bodySide: UploadFile = File(None),
    avatarStyle: str = Form("neutral"),
):
    """Last opp face + bodyFront (obligatorisk), bodySide (valgfritt). Returnerer jobId."""
    request_id = _get_request_id(request)
    try:
        print("AVATAR GENERATION START")
        print("Received files:", {"face": face, "bodyFront": bodyFront, "bodySide": bodySide})
        if not face and not bodyFront and not bodySide:
            print("NO FILES RECEIVED")

        if not face.content_type or not face.content_type.startswith("image/"):
            raise AppError(error_code="INPUT_INVALID", message="face must be an image", status_code=400)
        if not bodyFront.content_type or not bodyFront.content_type.startswith("image/"):
            raise AppError(error_code="INPUT_INVALID", message="bodyFront must be an image", status_code=400)
        style = (avatarStyle or "neutral").strip().lower()
        if style not in ("neutral", "male", "female"):
            raise AppError(
                error_code="INPUT_INVALID",
                message="Ugyldig avatarStyle. Bruk 'male', 'female' eller 'neutral'.",
                status_code=400,
                details={"request_id": request_id, "avatarStyle": avatarStyle},
            )

        job_id = str(uuid.uuid4())
        face_path = UPLOADS_DIR / f"{job_id}_face.jpg"
        body_front_path = UPLOADS_DIR / f"{job_id}_body_front.jpg"
        body_side_path = UPLOADS_DIR / f"{job_id}_body_side.jpg" if bodySide and bodySide.filename else None

        face_data = await face.read()
        body_front_data = await bodyFront.read()
        body_side_data = await bodySide.read() if body_side_path and bodySide else b""
        face_hash = hashlib.sha256(face_data).hexdigest()[:12] if face_data else None
        body_front_hash = hashlib.sha256(body_front_data).hexdigest()[:12] if body_front_data else None
        body_side_hash = hashlib.sha256(body_side_data).hexdigest()[:12] if body_side_data else None
        print(
            "UploadFile details:",
            {
                "face": {"filename": face.filename, "content_type": face.content_type, "size": len(face_data)},
                "bodyFront": {"filename": bodyFront.filename, "content_type": bodyFront.content_type, "size": len(body_front_data)},
                "bodySide": {
                    "filename": bodySide.filename if bodySide else None,
                    "content_type": bodySide.content_type if bodySide else None,
                    "size": len(body_side_data),
                },
            },
        )

        logger.info(
            "avatar/jobs upload request_id=%s avatarStyle=%s: face(name=%s, content_type=%s, size=%s, sha256_12=%s), bodyFront(name=%s, content_type=%s, size=%s, sha256_12=%s), bodySide(name=%s, content_type=%s, size=%s, sha256_12=%s)",
            request_id,
            style,
            face.filename,
            face.content_type,
            len(face_data),
            face_hash,
            bodyFront.filename,
            bodyFront.content_type,
            len(body_front_data),
            body_front_hash,
            bodySide.filename if bodySide else None,
            bodySide.content_type if bodySide else None,
            len(body_side_data),
            body_side_hash,
        )

        if not face_data and not body_front_data and not body_side_data:
            raise AppError(error_code="INPUT_INVALID", message="No images received", status_code=400)
        if not body_front_data:
            raise AppError(error_code="INPUT_INVALID", message="bodyFront mangler", status_code=400)

        # Robust decode for validation (supports more formats than cv2 alone).
        decoded = _decode_image_bytes(body_front_data)
        if decoded is None:
            heic_hint = (b"ftypheic" in body_front_data[:64]) or (b"ftypheif" in body_front_data[:64])
            raise AppError(
                error_code="INPUT_INVALID",
                message="bodyFront kunne ikke dekodes som bilde",
                status_code=400,
                details={"request_id": request_id, "format_hint": "heic/heif" if heic_hint else None},
                retryable=False,
            )
        h_body, w_body = decoded.shape[:2]
        if w_body < 160 or h_body < 220:
            raise AppError(
                error_code="INPUT_INVALID",
                message="Vennligst last opp full-body bilde med høyere oppløsning.",
                status_code=400,
                details={"width": int(w_body), "height": int(h_body), "min_width": 160, "min_height": 220},
            )

        face_path.write_bytes(face_data)
        body_front_path.write_bytes(body_front_data)
        if body_side_path and bodySide and body_side_data:
            body_side_path.write_bytes(body_side_data)

        # Debug copy should never break job creation.
        try:
            debug_main_path = None
            debug_face_path = None
            debug_body_front_path = None
            debug_body_side_path = None
            debug_main_path = DEBUG_INPUTS_DIR / f"{job_id}.jpg"
            debug_face_path = DEBUG_INPUTS_DIR / f"{job_id}_face_{Path(face.filename or 'face.jpg').name}"
            debug_body_front_path = DEBUG_INPUTS_DIR / f"{job_id}_body_front_{Path(bodyFront.filename or 'body_front.jpg').name}"
            debug_main_path.write_bytes(body_front_data)
            debug_face_path.write_bytes(face_data)
            debug_body_front_path.write_bytes(body_front_data)
            logger.info("debug input saved: %s", debug_main_path.resolve())
            logger.info("debug input saved: %s", debug_face_path.resolve())
            logger.info("debug input saved: %s", debug_body_front_path.resolve())
            if body_side_data:
                debug_body_side_path = DEBUG_INPUTS_DIR / f"{job_id}_body_side_{Path((bodySide.filename if bodySide else 'body_side.jpg')).name}"
                debug_body_side_path.write_bytes(body_side_data)
                logger.info("debug input saved: %s", debug_body_side_path.resolve())
        except Exception:
            logger.exception("debug input write failed request_id=%s job_id=%s", request_id, job_id)

        logger.info("stored upload path face: %s", face_path.resolve())
        logger.info("stored upload path bodyFront: %s", body_front_path.resolve())
        if body_side_path:
            logger.info("stored upload path bodySide: %s", body_side_path.resolve())
        logger.info(
            "stored upload hashes request_id=%s job_id=%s face_sha256_12=%s body_front_sha256_12=%s body_side_sha256_12=%s",
            request_id,
            job_id,
            face_hash,
            body_front_hash,
            body_side_hash,
        )

        _remember_upload(
            {
                "jobId": job_id,
                "requestId": request_id,
                "hashes": {
                    "face": face_hash,
                    "bodyFront": body_front_hash,
                    "bodySide": body_side_hash,
                },
                "debugInputPaths": {
                    "face": str(debug_face_path.resolve()) if debug_face_path else None,
                    "bodyFront": str(debug_body_front_path.resolve()) if debug_body_front_path else None,
                    "bodySide": str(debug_body_side_path.resolve()) if body_side_data and debug_body_side_path else None,
                },
            }
        )

        create_job(
            str(face_path),
            str(body_front_path),
            str(body_side_path) if body_side_path else None,
            job_id=job_id,
            avatar_style=style,
            request_id=request_id,
        )
        thread = threading.Thread(target=_run_job, args=(job_id, request_id))
        thread.start()
        return JobCreateResponse(jobId=job_id)
    except AppError:
        raise
    except Exception as e:
        logger.exception("create_avatar_job failed request_id=%s", request_id)
        raise AppError(
            error_code="JOB_CREATE_FAILED",
            message="Kunne ikke starte avatar-generering.",
            status_code=500,
            details={"request_id": request_id, "exception": type(e).__name__},
            retryable=True,
        ) from e


@app.get("/avatar/jobs/{job_id}", response_model=JobStatusResponse)
def get_avatar_job(job_id: str):
    """Hent status for en avatar-job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    base_url = ""  # client will use same origin or config
    avatar_url = job.avatar_url
    if avatar_url and not avatar_url.startswith("http") and base_url:
        avatar_url = base_url.rstrip("/") + avatar_url
    return JobStatusResponse(
        status=job.status,
        progress=job.progress,
        progress_message=job.progress_message,
        avatarUrl=avatar_url,
        bodyDebug=sanitize_for_json(job.body_debug) if job.body_debug else None,
        error=job.error,
        message=job.error,
        error_code=job.error_code,
        request_id=job.request_id,
        retryable=job.retryable,
        details=sanitize_for_json(job.error_details) if (is_dev_mode() and job.error_details) else None,
        avatarStyle=job.avatar_style,
    )


# --- Legacy single-shot avatar (PNG) ---

class AvatarResponse(BaseModel):
    success: bool
    avatar_base64: str | None = None
    error: str | None = None


@app.post("/avatar", response_model=AvatarResponse)
async def create_avatar(file: UploadFile = File(...)):
    """Last opp fjesbilde, få tilbake avatar som base64 PNG (legacy)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Kun bilder tillatt")
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return AvatarResponse(success=False, error="Kunne ikke lese bildet")
        face = extract_face(img)
        if face is None:
            return AvatarResponse(
                success=False,
                error="Ingen fjes funnet. Prøv et tydeligere selfie.",
            )
        avatar_bytes = make_avatar(face)
        b64 = base64.b64encode(avatar_bytes).decode()
        return AvatarResponse(success=True, avatar_base64=b64)
    except Exception as e:
        logger.exception("legacy /avatar failed")
        raise AppError(
            error_code="INTERNAL_ERROR",
            message="Legacy avatar-endepunkt feilet.",
            status_code=500,
            details={"raw_error": str(e)},
            retryable=False,
        ) from e


@app.get("/health")
def health():
    return {"ok": True}
