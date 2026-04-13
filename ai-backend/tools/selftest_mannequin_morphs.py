from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import debug_mannequin_morphs  # pylint: disable=import-error


def main() -> int:
    failures = []
    report = {}
    for style in ("female", "male", "neutral"):
        dbg = debug_mannequin_morphs(style)
        report[style] = dbg
        if not dbg.get("exists"):
            failures.append(f"{style}: base model missing")
            continue
        if not dbg.get("hasMorphTargets"):
            failures.append(f"{style}: morph targets missing")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        print("\nSELFTEST FAILED:")
        for f in failures:
            print(f"- {f}")
        return 1
    print("\nSELFTEST OK: mannequin morph targets exist for all styles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
