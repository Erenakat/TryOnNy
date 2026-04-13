from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


def _env_is_true(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def is_dev_mode() -> bool:
    return _env_is_true("DEV_ERROR_DETAILS", "1")


@dataclass
class AppError(Exception):
    error_code: str
    message: str
    status_code: int = 500
    details: Optional[dict[str, Any]] = None
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


def sanitize_for_json(value: Any) -> Any:
    """
    Best-effort conversion of arbitrary python objects into JSON-serializable types.
    This prevents FastAPI response serialization from crashing and turning a real error
    into a generic 500 "INTERNAL_ERROR".
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Numpy scalars
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v) for v in value]
    # Path-like or other objects
    return str(value)


def error_response_payload(
    *,
    error_code: str,
    message: str,
    request_id: str,
    retryable: bool = False,
    details: Optional[dict[str, Any]] = None,
    include_details: Optional[bool] = None,
) -> dict[str, Any]:
    if include_details is None:
        include_details = is_dev_mode()
    payload: dict[str, Any] = {
        "status": "failed",
        "error_code": error_code,
        "message": message,
        "request_id": request_id,
        "retryable": retryable,
    }
    if include_details and details:
        payload["details"] = sanitize_for_json(details)
    return payload


def classify_pose_exception(exc: Exception) -> AppError:
    msg = str(exc)
    lower = msg.lower()

    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return AppError(
            error_code="POSE_DEPENDENCY_MISSING",
            message="Server mangler pose-komponenter.",
            status_code=503,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=False,
        )
    if isinstance(exc, TimeoutError):
        return AppError(
            error_code="POSE_TIMEOUT",
            message="Kroppsanalyse tidsavbrudd på serveren.",
            status_code=503,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=True,
        )
    if "cuda" in lower or "out of memory" in lower or "oom" in lower:
        return AppError(
            error_code="POSE_OOM",
            message="Serveren er midlertidig tom for minne under kroppsanalyse.",
            status_code=503,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=True,
        )
    if msg.startswith("POSE_NO_KEYPOINTS:") or "no keypoints detected" in lower:
        return AppError(
            error_code="POSE_NO_KEYPOINTS",
            message="Fant ikke tydelige kroppspunkter i bildet.",
            status_code=422,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=False,
        )
    if msg.startswith("POSE_DEPENDENCY_MISSING:"):
        return AppError(
            error_code="POSE_DEPENDENCY_MISSING",
            message="Server mangler pose-komponenter.",
            status_code=503,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=False,
        )
    if msg.startswith("POSE_MODEL_INIT_FAILED:") or msg.startswith("POSE_MODEL_LOAD_FAILED:"):
        return AppError(
            error_code="POSE_MODEL_INIT_FAILED",
            message="Kunne ikke initialisere pose-modellen.",
            status_code=503,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=True,
        )
    if msg.startswith("INPUT_INVALID:"):
        return AppError(
            error_code="INPUT_INVALID",
            message=msg.split("INPUT_INVALID:", 1)[1].strip() or "Ugyldig input.",
            status_code=400,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=False,
        )
    if msg.startswith("POSE_FAILED:") or msg.startswith("POSE_INCOMPLETE_KEYPOINTS:"):
        return AppError(
            error_code="POSE_INFERENCE_FAILED",
            message="Kroppsanalyse feilet under estimering.",
            status_code=503,
            details={"exception": type(exc).__name__, "raw_error": msg},
            retryable=True,
        )
    return AppError(
        error_code="POSE_INFERENCE_FAILED",
        message="Uventet feil under kroppsanalyse.",
        status_code=503,
        details={"exception": type(exc).__name__, "raw_error": msg},
        retryable=True,
    )
