from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow importing from ai-backend (folder name contains a dash).
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "ai-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pipeline import estimate_skin_color_from_face, apply_skin_color_to_glb  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate skin color from face image and apply to a GLB.")
    parser.add_argument("--face", required=True, help="Path to face image (jpg/png/heic if supported)")
    parser.add_argument("--glb", required=True, help="Path to input .glb")
    parser.add_argument("--out", default=None, help="Optional output .glb path (default: <glb>_skin.glb)")
    args = parser.parse_args()

    face_p = Path(args.face).expanduser().resolve()
    glb_p = Path(args.glb).expanduser().resolve()
    if not face_p.exists():
        print("face not found:", str(face_p))
        return 2
    if not glb_p.exists():
        print("glb not found:", str(glb_p))
        return 2

    out_p = Path(args.out).expanduser().resolve() if args.out else glb_p.with_name(glb_p.stem + "_skin.glb")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(glb_p, out_p)

    srgb, dbg = estimate_skin_color_from_face(str(face_p))
    print("face_bbox:", dbg.get("face_bbox"))
    print("method:", dbg.get("method"))
    print("pixels_used:", dbg.get("pixels_used"))
    print("rgb8:", dbg.get("rgb8"))
    print("srgb_factor:", list(srgb))

    res = apply_skin_color_to_glb(str(out_p), list(srgb), request_id="cli-test-skin")
    print("applied:", res.get("applied"))
    print("materials:", res.get("materials"))
    print("out:", str(out_p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

