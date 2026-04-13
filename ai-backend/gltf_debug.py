from __future__ import annotations

from pathlib import Path
from typing import Any


def gltf_debug_morphs(glb_path: str) -> dict[str, Any]:
    """
    Return morph/mesh debug information for a GLB/GLTF file.
    Never raises; always returns a JSON-safe dict.
    """
    out: dict[str, Any] = {
        "ok": False,
        "path": str(glb_path),
        "exists": False,
        "hasMorphTargets": False,
        "meshCount": 0,
        "meshesWithMorphTargets": 0,
        "totalMorphTargets": 0,
        "meshMorphCounts": {},
        "meshMorphNames": {},
        "nodeWeightsPresent": False,
        "meshWeightsPresent": False,
        "hasUnderwearMesh": False,
        "underwearMeshes": [],
        "error": None,
    }
    try:
        from pygltflib import GLTF2
    except Exception:
        out["error"] = "pygltflib_unavailable"
        return out

    p = Path(glb_path)
    out["exists"] = bool(p.exists())
    if not p.exists():
        out["error"] = "file_missing"
        return out

    try:
        gltf = GLTF2().load(str(p))
        meshes = gltf.meshes or []
        out["meshCount"] = int(len(meshes))
        underwear_kws = ("underwear", "bikini", "bra", "panty", "brief", "boxer", "shorts", "swim")
        uw: list[str] = []

        mesh_morph_counts: dict[str, int] = {}
        mesh_morph_names: dict[str, list[str]] = {}
        meshes_with_targets = 0
        total_targets = 0

        for mi, mesh in enumerate(meshes):
            mesh_name = str(getattr(mesh, "name", None) or f"mesh_{mi}")
            low = mesh_name.lower()
            if any(k in low for k in underwear_kws):
                uw.append(mesh_name)

            target_count = 0
            target_names: list[str] = []
            extras = getattr(mesh, "extras", None)
            if isinstance(extras, dict):
                names = extras.get("targetNames")
                if isinstance(names, list):
                    target_names = [str(x) for x in names]

            for prim in (getattr(mesh, "primitives", None) or []):
                targets = getattr(prim, "targets", None)
                if targets:
                    target_count = max(target_count, int(len(targets)))

            if target_count > len(target_names):
                target_names.extend([f"target_{i}" for i in range(len(target_names), target_count)])
            target_names = target_names[:target_count]

            mesh_morph_counts[mesh_name] = int(target_count)
            mesh_morph_names[mesh_name] = list(target_names)

            if target_count > 0:
                meshes_with_targets += 1
                total_targets += int(target_count)

        node_weights = any(getattr(n, "weights", None) is not None for n in (gltf.nodes or []))
        mesh_weights = any(getattr(m, "weights", None) is not None for m in meshes)

        out["ok"] = True
        out["meshMorphCounts"] = mesh_morph_counts
        out["meshMorphNames"] = mesh_morph_names
        out["meshesWithMorphTargets"] = int(meshes_with_targets)
        out["totalMorphTargets"] = int(total_targets)
        out["hasMorphTargets"] = bool(total_targets > 0)
        out["nodeWeightsPresent"] = bool(node_weights)
        out["meshWeightsPresent"] = bool(mesh_weights)
        out["hasUnderwearMesh"] = bool(len(uw) > 0)
        out["underwearMeshes"] = sorted(set(uw))
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
