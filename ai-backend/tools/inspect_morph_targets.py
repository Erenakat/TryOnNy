from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_morph_targets.py <path-to-avatar.glb>")
        return 1

    glb_path = Path(sys.argv[1]).resolve()
    if not glb_path.exists():
        print(json.dumps({"ok": False, "error": "file_not_found", "path": str(glb_path)}, ensure_ascii=False, indent=2))
        return 2

    # Import from backend root.
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from face_likeness import inspect_head_morph_targets  # pylint: disable=import-error

    report = inspect_head_morph_targets(str(glb_path))
    print(json.dumps({"ok": True, "path": str(glb_path), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
