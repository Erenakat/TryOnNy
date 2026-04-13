"""
Download base mannequin GLB. Required - no procedural avatar generation.
Source asset: GDQuest "Mannequiny" (neutral humanoid, A-pose).
License: CC BY 4.0 for art assets (free with attribution).
Run: python download_mannequin.py
"""
import logging
import urllib.request
from pathlib import Path

LOG = logging.getLogger(__name__)
OUT_DIR = Path(__file__).resolve().parent / "static" / "avatars"
OUT_FILE = OUT_DIR / "base_mannequin.glb"

# GDQuest mannequin (free art asset, CC BY 4.0)
MANNEQUIN_URL = (
    "https://raw.githubusercontent.com/gdquest-demos/godot-3d-mannequin/"
    "master/godot-csharp/assets/3d/mannequiny/mannequiny-0.3.0.glb"
)


def download() -> bool:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        LOG.info("Downloading mannequin from %s", MANNEQUIN_URL)
        req = urllib.request.Request(MANNEQUIN_URL, headers={"User-Agent": "TryOn-Avatar/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        OUT_FILE.write_bytes(data)
        LOG.info("Saved to %s (%s bytes)", OUT_FILE, len(data))
        return True
    except Exception as e:
        LOG.exception("Download failed: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = download()
    exit(0 if ok else 1)
