from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Allow importing from ai-backend (folder name contains a dash).
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "ai-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from errors import AppError  # type: ignore
from pipeline import run_pipeline  # type: ignore


def _format_hint(data: bytes) -> str | None:
    if not data:
        return None
    head = data[:64]
    if b"ftypheic" in head or b"ftypheif" in head:
        return "heic/heif"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] == b"RIFF" and b"WEBP" in head[:16]:
        return "webp"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run avatar pipeline locally for debugging")
    parser.add_argument("--face", required=True, help="Path to face image")
    parser.add_argument("--front", required=True, help="Path to body front image")
    parser.add_argument("--side", default=None, help="Optional body side image")
    parser.add_argument("--out", required=True, help="Output .glb path")
    args = parser.parse_args()

    face_p = Path(args.face).expanduser().resolve()
    front_p = Path(args.front).expanduser().resolve()
    side_p = Path(args.side).expanduser().resolve() if args.side else None
    out_p = Path(args.out).expanduser().resolve()

    missing = []
    if not face_p.exists():
        missing.append(f"--face not found: {face_p}")
    if not front_p.exists():
        missing.append(f"--front not found: {front_p}")
    if side_p and not side_p.exists():
        missing.append(f"--side not found: {side_p}")
    if missing:
        print("\n".join(missing))
        print("Hint: use absolute paths or quote paths with spaces.")
        return 2

    face = str(face_p)
    front = str(front_p)
    side = str(side_p) if side_p else None
    out = str(out_p)

    # Run pipeline with a stable request_id so logs are searchable.
    request_id = "cli-test-pipeline"
    try:
        debug = run_pipeline(face, front, side, out, request_id=request_id)
    except AppError as e:
        print("request_id:", request_id)
        print("error_code:", e.error_code)
        print("message:", e.message)
        print("details:", e.details)
        try:
            fb = face_p.read_bytes()
            print("face_bytes:", len(fb), "format_hint:", _format_hint(fb))
        except Exception:
            pass
        try:
            bb = front_p.read_bytes()
            print("front_bytes:", len(bb), "format_hint:", _format_hint(bb))
        except Exception:
            pass
        return 1

    # Print requested diagnostics.
    # Face bbox: recompute via exported helper using the images the pipeline already read (best-effort).
    # We can infer bbox presence from the logs too, but print here if present in debug payload.
    print("request_id:", request_id)
    print("face_bbox_found:", "unknown (see logs)")  # bbox is logged in pipeline; keep CLI stable
    print("skin_rgb:", debug.get("skinColorRgb"))
    print("skin_linear:", debug.get("skinColorLinear"))
    print("skin_materials_updated:", debug.get("skinMaterialsUpdated"))
    print("face_texture_applied:", debug.get("faceTextureApplied"))
    print("face_materials_updated:", debug.get("faceMaterialsUpdated"))

    data = out_p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    print("out_path:", str(out_p))
    print("out_size_bytes:", len(data))
    print("out_sha256:", sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

