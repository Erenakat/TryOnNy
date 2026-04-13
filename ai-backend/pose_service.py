from __future__ import annotations

import importlib
import logging
import os
import traceback
from types import ModuleType
from typing import Any, Optional

from errors import AppError

logger = logging.getLogger(__name__)


class PoseService:
    def __init__(self) -> None:
        self.detector: Any = None
        self.init_error: Optional[Exception] = None
        self.initialized: bool = False
        self.model_name: str = "mediapipe_pose"
        self.mp_info: dict[str, Any] = {}
        self.init_error_stacktrace: Optional[str] = None

    def _import_dependency(self, module_name: str) -> ModuleType:
        return importlib.import_module(module_name)

    def _missing_module_name(self, exc: Exception) -> str:
        if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", None):
            return str(exc.name)
        return str(exc)

    def dependency_status(self) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for name in ("mediapipe", "cv2", "numpy"):
            try:
                mod = self._import_dependency(name)
                if name == "mediapipe":
                    # mediapipe must expose either Solutions or Tasks pose API
                    has_solutions_pose = False
                    try:
                        has_solutions_pose = bool(
                            hasattr(mod, "solutions")
                            and getattr(mod, "solutions", None) is not None
                            and hasattr(getattr(mod.solutions, "pose", None), "Pose")
                        )
                    except Exception:
                        has_solutions_pose = False
                    has_tasks_pose = False
                    try:
                        from mediapipe.tasks.python.vision import PoseLandmarker  # noqa: F401
                        has_tasks_pose = True
                    except Exception:
                        has_tasks_pose = False
                    status[name] = bool(has_solutions_pose or has_tasks_pose)
                else:
                    status[name] = True
            except Exception:
                status[name] = False
        return status

    def init(self) -> bool:
        if self.initialized and self.detector is not None:
            return True
        try:
            self._import_dependency("cv2")
            self._import_dependency("numpy")
            mp = self._import_dependency("mediapipe")
        except ModuleNotFoundError as e:
            self.detector = None
            self.init_error = e
            self.initialized = True
            logger.exception("Pose detector init failed: missing dependency: %s", self._missing_module_name(e))
            return False
        except Exception as e:
            self.detector = None
            self.init_error = e
            self.initialized = True
            logger.exception("Pose detector init failed while importing dependencies")
            return False

        try:
            mp_file = getattr(mp, "__file__", None)
            mp_ver = getattr(mp, "__version__", None)
            logger.info("mediapipe import: file=%s version=%s has_solutions=%s", mp_file, mp_ver, hasattr(mp, "solutions"))
            self.mp_info = {
                "file": str(mp_file) if mp_file is not None else None,
                "version": str(mp_ver) if mp_ver is not None else None,
                "has_solutions": bool(hasattr(mp, "solutions")),
            }

            # Detect common "wrong mediapipe" import (namespace conflict / shadowing).
            # Note: venvs may live inside the repo (ai-backend\.venv\...), so we must NOT
            # treat "inside cwd" as a conflict. A real conflict is typically a local
            # mediapipe.py or mediapipe/ package that is NOT from site-packages.
            if mp_file:
                mp_file_l = str(mp_file).replace("\\", "/").lower()
                if ("site-packages" not in mp_file_l) and ("dist-packages" not in mp_file_l):
                    raise RuntimeError(f"mediapipe import conflict (not from site-packages): {mp_file}")

            pose_cls = None
            if hasattr(mp, "solutions") and getattr(mp, "solutions", None) is not None:
                try:
                    pose_mod = getattr(mp.solutions, "pose", None)
                    pose_cls = getattr(pose_mod, "Pose", None) if pose_mod is not None else None
                except Exception:
                    pose_cls = None

            if pose_cls is None:
                # Some installs may not expose mp.solutions.pose as attribute; try direct import.
                try:
                    pose_mod = importlib.import_module("mediapipe.solutions.pose")
                    pose_cls = getattr(pose_mod, "Pose", None)
                except Exception:
                    pose_cls = None

            # Primary backend: Solutions Pose (fast, simple)
            if pose_cls is not None:
                try:
                    self.detector = pose_cls(
                        static_image_mode=True,
                        model_complexity=2,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.model_name = "mediapipe.solutions.pose"
                    self.init_error = None
                    self.init_error_stacktrace = None
                    self.initialized = True
                    logger.info("Pose detector initialized successfully (backend=%s)", self.model_name)
                    return True
                except Exception as e:
                    logger.exception("mediapipe.solutions.pose init failed; trying tasks PoseLandmarker fallback")

            # Fallback backend: mediapipe.tasks PoseLandmarker using downloaded .task model
            try:
                from models_download import get_pose_model_path
                from mediapipe.tasks.python.core import base_options
                from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
                from mediapipe.tasks.python.vision.core import vision_task_running_mode

                model_path = get_pose_model_path()
                opts = PoseLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=model_path),
                    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                    num_poses=1,
                )

                landmarker = PoseLandmarker.create_from_options(opts)

                class _TasksPoseAdapter:
                    def __init__(self, lm):
                        self._lm = lm

                    def process(self, rgb_np):
                        # rgb_np is uint8 RGB (H,W,3)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
                        return self._lm.detect(mp_image)

                self.detector = _TasksPoseAdapter(landmarker)
                self.model_name = f"mediapipe.tasks.pose_landmarker (model={model_path})"
                self.init_error = None
                self.init_error_stacktrace = None
                self.initialized = True
                logger.info("Pose detector initialized successfully (backend=%s)", self.model_name)
                return True
            except Exception as e:
                raise RuntimeError(f"PoseLandmarker fallback init failed: {e}") from e
        except Exception as e:
            self.detector = None
            self.init_error = e
            self.init_error_stacktrace = traceback.format_exc()
            self.initialized = True
            logger.exception("Pose detector init failed while creating mediapipe pose")
            return False

    def get_detector(self, request_id: Optional[str] = None):
        if not self.initialized:
            self.init()
        if self.detector is not None:
            return self.detector

        init_error = self.init_error
        if isinstance(init_error, ModuleNotFoundError):
            missing = self._missing_module_name(init_error)
            raise AppError(
                error_code="POSE_DEPENDENCY_MISSING",
                message=f"Pose dependency mangler: {missing}",
                status_code=503,
                details={
                    "missing": missing,
                    "request_id": request_id,
                    "init_error": str(init_error),
                    "mediapipe": dict(self.mp_info),
                },
                retryable=False,
            )
        raise AppError(
            error_code="POSE_MODEL_INIT_FAILED",
            message="Kunne ikke initialisere pose-detector.",
            status_code=503,
            details={
                "request_id": request_id,
                "init_error_type": type(init_error).__name__ if init_error else None,
                "init_error": str(init_error) if init_error else "unknown init error",
                "mediapipe": dict(self.mp_info),
                "backend": self.model_name,
                "stacktrace": self.init_error_stacktrace,
            },
            retryable=True,
        )

    def health(self) -> dict[str, Any]:
        deps = self.dependency_status()
        available = self.detector is not None
        error_code = None
        error_message = None
        if not available:
            if isinstance(self.init_error, ModuleNotFoundError):
                error_code = "POSE_DEPENDENCY_MISSING"
                error_message = self._missing_module_name(self.init_error)
            elif self.initialized:
                error_code = "POSE_MODEL_INIT_FAILED"
                error_message = str(self.init_error) if self.init_error else "unknown init error"
        return {
            "available": available,
            "error_code": error_code,
            "error_message": error_message,
            "deps": {"mediapipe": deps.get("mediapipe", False), "cv2": deps.get("cv2", False)},
            "model": self.model_name,
            "initialized": self.initialized,
            "mediapipe": dict(self.mp_info),
            "init_error_type": type(self.init_error).__name__ if self.init_error else None,
        }


pose_service = PoseService()
