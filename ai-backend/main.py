"""
AI Avatar Backend – avatar fra fjes + kropp (job-basert, GLB).
Endpoints: POST /avatar/jobs, GET /avatar/jobs/:jobId
Bruk: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import io
import json
import re
import base64
import uuid
import threading
import logging
import hashlib
import traceback
import os
import time
import math
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
from openai import OpenAI, BadRequestError
from PIL import Image
from dotenv import load_dotenv

from jobs import create_job, get_job, update_job, JobStatus
from pipeline import run_pipeline, get_face_region, ensure_models_loaded, _decode_image_bytes, debug_mannequin_morphs  # type: ignore
from errors import AppError, error_response_payload, is_dev_mode, classify_pose_exception, sanitize_for_json
from pose_service import pose_service
from product_feed import load_products

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

# Load local .env from ai-backend/ regardless of cwd. Safe no-op if missing.
load_dotenv(dotenv_path=BASE_DIR / ".env")

UPLOADS_DIR = BASE_DIR / "uploads"
AVATARS_DIR = BASE_DIR / "static" / "avatars"
DEBUG_INPUTS_DIR = BASE_DIR / "static" / "debug_inputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

_recent_errors_by_request_id: dict[str, dict] = {}
_recent_uploads: deque[dict] = deque(maxlen=200)


class ColorChatRequest(BaseModel):
    message: str
    rgb: list[int] | None = None
    hex: str | None = None
    source: str | None = None
    history: list[dict] | None = None
    # Optional: allow client to send key (dev-only / self-hosted setups).
    api_key: str | None = None


def _get_openai_api_key(request: Request | None = None, payload_api_key: str | None = None) -> str | None:
    """
    Priority:
      1) Request header: x-openai-key
      2) Payload field: api_key
      3) Environment: OPENAI_API_KEY / OPENAI_APIKEY
    """
    if request is not None:
        hdr = request.headers.get("x-openai-key") or request.headers.get("X-OpenAI-Key")
        if hdr and isinstance(hdr, str) and hdr.strip():
            return hdr.strip()
    if payload_api_key and isinstance(payload_api_key, str) and payload_api_key.strip():
        return payload_api_key.strip()
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")


def _require_openai_api_key(api_key: str | None) -> str:
    if api_key and isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    raise HTTPException(
        status_code=400,
        detail="Mangler OpenAI API key. Send `x-openai-key` (header) / `apiKey` (form) eller sett `OPENAI_API_KEY` på serveren.",
    )


def _parse_hex_color(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except Exception:
        return None


def _rgb_dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _extract_best_hexes(analysis: dict) -> list[str]:
    try:
        colors = analysis.get("best_colors")
        if not isinstance(colors, list):
            return []
        out: list[str] = []
        for c in colors:
            if not isinstance(c, dict):
                continue
            hx = c.get("hex")
            if not isinstance(hx, str):
                continue
            hx = hx.strip()
            if not hx:
                continue
            if not hx.startswith("#"):
                hx = "#" + hx
            out.append(hx.upper())
        seen = set()
        uniq = []
        for hx in out:
            if hx in seen:
                continue
            seen.add(hx)
            uniq.append(hx)
        return uniq
    except Exception:
        return []


def _rank_products_by_palette(products: list[dict], best_hexes: list[str], limit: int) -> list[dict]:
    limit = max(1, min(int(limit or 12), 50))
    if not products:
        return []
    if not best_hexes:
        return products[:limit]

    palette_rgb = [(_parse_hex_color(hx), hx) for hx in best_hexes]
    palette_rgb = [(rgb, hx) for (rgb, hx) in palette_rgb if rgb is not None]
    if not palette_rgb:
        return products[:limit]

    scored: list[tuple[float, dict]] = []
    for p in products:
        prgb = _parse_hex_color(p.get("color_hex"))
        if prgb is None:
            scored.append((1e9, p))
            continue
        d = min(_rgb_dist(prgb, rgb) for (rgb, _) in palette_rgb)
        scored.append((d, p))

    scored.sort(key=lambda t: t[0])

    picked: list[dict] = []
    used_titles: set[str] = set()
    used_categories: set[str] = set()
    used_hex: set[str] = set()

    for _, p in scored:
        if len(picked) >= limit:
            break
        title = str(p.get("title") or "").strip()
        if title and title in used_titles:
            continue
        cat = str(p.get("category") or "").strip().lower()
        hx = str(p.get("color_hex") or "").strip().upper()
        if hx and not hx.startswith("#"):
            hx = "#" + hx

        if len(products) >= 8:
            if cat and cat in used_categories and len(used_categories) < 4:
                continue
            if hx and hx in used_hex and len(used_hex) < 6:
                continue

        picked.append(p)
        if title:
            used_titles.add(title)
        if cat:
            used_categories.add(cat)
        if hx:
            used_hex.add(hx)

    if len(picked) < limit:
        for _, p in scored:
            if len(picked) >= limit:
                break
            if p in picked:
                continue
            picked.append(p)

    return picked


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


@app.post("/color/chat")
async def color_chat(payload: ColorChatRequest, request: Request):
    """
    Proxy for "Fargeanalyse"-chat. Reads OPENAI_API_KEY from environment.
    Returns: { reply: string }
    """
    request_id = _get_request_id(request)
    api_key = _require_openai_api_key(_get_openai_api_key(request, payload.api_key))

    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Mangler message.")

    client = OpenAI(api_key=api_key)

    color_ctx = {
        "rgb": payload.rgb,
        "hex": payload.hex,
        "source": payload.source,
    }

    system = (
        "Du er en hjelpsom stylist og fargeanalytiker. "
        "Svar på norsk. Vær konkret og kort. "
        "Gi forslag til farger som passer (klær, hår, sminke) basert på hudtone-data når tilgjengelig. "
        "Hvis data mangler, be om et tydelig fjes + kropp-bilde og foreslå neste steg."
    )

    messages = [{"role": "system", "content": system}]
    if payload.history and isinstance(payload.history, list):
        # Expect items like {role, content}. Keep it small.
        for item in payload.history[-8:]:
            try:
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content.strip()[:1500]})
            except Exception:
                continue

    if any(v is not None for v in color_ctx.values()):
        messages.append(
            {
                "role": "user",
                "content": f"Kontekst (fra app): {sanitize_for_json(color_ctx)}",
            }
        )

    messages.append({"role": "user", "content": msg[:2000]})

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.4,
        )
        reply = (resp.choices[0].message.content or "").strip()
        if not reply:
            reply = "Jeg fikk ikke noe svar tilbake. Prøv igjen."
        return {"reply": reply, "request_id": request_id}
    except Exception as e:
        logger.exception("color_chat failed request_id=%s", request_id)
        _remember_error(
            request_id,
            {"error_code": "COLOR_CHAT_FAILED", "message": "color chat failed", "details": {"exception": str(e)}},
        )
        raise HTTPException(status_code=502, detail="Fargeanalyse-chat feilet. Sjekk server-loggene.")


class ColorAnalyzeResponse(BaseModel):
    request_id: str
    analysis: dict
    products: list[dict]


def _parse_json_object_from_llm(text: str) -> dict:
    """OpenAI returnerer ofte JSON omgitt av markdown; trekk ut objektet vi trenger."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Tomt svar fra modellen.")
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = None
    if isinstance(out, dict):
        return out
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if m:
        try:
            out = json.loads(m.group(1).strip())
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            out = json.loads(raw[start : end + 1])
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
    raise ValueError("Kunne ikke parse JSON-objekt fra modellens svar.")


def _bytes_for_openai_vision(img_bytes: bytes, content_type: str | None) -> tuple[bytes, str]:
    """
    OpenAI Vision støtter pålitelig JPEG/PNG/GIF/WebP i data-URL.
    HEIC/HEIF fra iOS (og andre rare MIME-typer) dekodes til JPEG via Pillow + pillow-heif.
    """
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct == "image/jpg":
        ct = "image/jpeg"
    if ct in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        return img_bytes, ct

    try:
        try:
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        im = Image.open(io.BytesIO(img_bytes))
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception as exc:
        extra = ""
        if ct in ("image/heic", "image/heif"):
            extra = " Sørg for at pillow-heif er installert på serveren, eller eksporter bildet som JPEG."
        raise ValueError(f"Kunne ikke forberede bildet for AI (type={ct or 'ukjent'}).{extra}") from exc


@app.post("/color/analyze-image", response_model=ColorAnalyzeResponse)
async def color_analyze_image(
    request: Request,
    image: UploadFile = File(...),
    apiKey: str | None = Form(None),
    productLimit: int = Form(12),
):
    """
    Analyser et bilde (f.eks. ansikt/selfie) med OpenAI og returner strukturert fargeanalyse
    + konkrete produktforslag (med image_url) fra lokal feed.

    API-nøkkel kan sendes som header `x-openai-key` eller form-felt `apiKey`.
    """
    request_id = _get_request_id(request)
    api_key = _require_openai_api_key(_get_openai_api_key(request, apiKey))

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image må være et bilde.")

    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Tomt bilde.")

    model = os.environ.get("OPENAI_MODEL_VISION", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    client = OpenAI(api_key=api_key)

    # Keep prompt tight and deterministic: return JSON only.
    system = (
        "Du er en ekspert på fargeanalyse for klær. "
        "Svar på norsk. Returner KUN gyldig JSON (ingen markdown). "
        "Ikke inkluder personidentifiserende info."
    )
    user = (
        "Analyser hudtone/undertone og foreslå en sesong (vår/sommer/høst/vinter) basert på synlige trekk i bildet. "
        "VIKTIG: Unngå 'default'-svar. Ikke velg 'autumn' med mindre det er tydelige varme/jordnære trekk. "
        "Sørg for variasjon: ikke gjenta samme farge overalt.\n\n"
        "Returner et JSON-objekt med nøkler:\n"
        "- undertone: 'warm'|'cool'|'neutral'\n"
        "- depth: 'light'|'medium'|'deep'\n"
        "- contrast: 'low'|'medium'|'high'\n"
        "- season: 'spring'|'summer'|'autumn'|'winter'\n"
        "- subseason: en av følgende (12-sesong):\n"
        "  - spring: 'light_spring'|'true_spring'|'bright_spring'\n"
        "  - summer: 'light_summer'|'true_summer'|'soft_summer'\n"
        "  - autumn: 'soft_autumn'|'true_autumn'|'deep_autumn'\n"
        "  - winter: 'bright_winter'|'true_winter'|'deep_winter'\n"
        "- season_reason: kort tekst\n"
        "- season_confidence: tall 0.0-1.0\n"
        "- best_colors: array (5-7 stk) av {name, hex} med UNIKE hex-verdier\n"
        "- avoid_colors: array (3-5 stk) av {name, hex} med UNIKE hex-verdier\n"
        "- notes: array (8-14 stk) av konkrete, nyttige punkt (1 setning hver). Inkluder tips om materialer, mønstre, kontraster, smykker/tilbehør, og “hva du bør prioritere”.\n"
        "- recommended_outfits: {everyday: array, formal: array}\n"
        "  - everyday: 4 plagg (overdel, bukse, sko, tilbehør) med UNIKE farger\n"
        "  - formal: 4 plagg (jakke, skjorte, bukse, sko) med UNIKE farger\n"
        "  - Hvert element: {item, color_name, color_hex}\n"
        "  - Bruk farger fra best_colors (eller nært beslektet), ikke samme hex på flere elementer.\n"
    )

    try:
        vision_bytes, vision_mime = _bytes_for_openai_vision(img_bytes, image.content_type)
        data_url = "data:" + vision_mime + ";base64," + base64.b64encode(vision_bytes).decode("utf-8")
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
        except BadRequestError:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
            )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Tomt svar fra OpenAI.")
        analysis = sanitize_for_json(_parse_json_object_from_llm(text))
    except Exception as e:
        logger.exception("color_analyze_image failed request_id=%s", request_id)
        _remember_error(
            request_id,
            {
                "error_code": "COLOR_ANALYZE_FAILED",
                "message": "color analyze failed",
                "details": {"exception": type(e).__name__, "exception_message": str(e)},
            },
        )
        detail = "Bildeanalyse feilet. Sjekk server-loggene."
        if is_dev_mode():
            detail = f"{detail} ({type(e).__name__}: {str(e)})"
        raise HTTPException(status_code=502, detail=detail)

    # Attach products matched to palette.
    raw_products = load_products(BASE_DIR, limit=60)
    best_hexes = _extract_best_hexes(analysis if isinstance(analysis, dict) else {})
    products = _rank_products_by_palette(raw_products, best_hexes, limit=int(productLimit or 12))
    return {"request_id": request_id, "analysis": analysis, "products": products}


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
