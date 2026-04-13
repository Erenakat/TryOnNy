"""
Generate default avatar GLB for fallback/preview.
Run from ai-backend: python generate_default_avatar.py
Output: static/avatars/default.glb
Uses base mannequin (auto-downloaded if missing).
"""
from pathlib import Path

from pipeline import _load_base_mannequin_scene, _ensure_base_mannequin

OUT_PATH = Path(__file__).resolve().parent / "static" / "avatars" / "default.glb"

def main():
    skin_rgb = (75, 65, 60)  # mørk figur
    _ensure_base_mannequin()
    scene = _load_base_mannequin_scene(skin_rgb, None, None)
    if scene is None:
        raise SystemExit(
            "Mannequin required. Run: python download_mannequin.py"
        )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(OUT_PATH), file_type="glb")
    print(f"Wrote {OUT_PATH}")
