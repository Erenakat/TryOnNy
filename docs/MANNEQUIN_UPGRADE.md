# Mannequin Avatar Upgrade

## Base mannequin (påkrevd)

Kroppen er nå en importert humanoid mannequin – ingen sfærer, kapsler eller primitiver.

**Første gangs oppsett:**
```bash
cd ai-backend
python download_mannequin.py   # nedlaster CesiumMan GLB (Khronos, CC0-lignende)
python generate_default_avatar.py
```

Pipelinen laster ned mannequinen automatisk ved første generering hvis den mangler.

## Endringer

### 1. 3D-mesh (pipeline.py)
- **Avrundet geometri:** Torso bygges av 3 kapsler (chest, waist, hips) i stedet for boks
- **Lemmer:** Kapsler med rundede ender i stedet for sylindere
- **Hender:** Enkle sfærer ved håndledd
- **Føtter:** Avflatede ellipsoider ved ankler
- **Hals:** Sylinder med riktig proporsjon (clavicle-hint)
- **CYL_SECTIONS:** 12 for glattere silhuett

### 2. PBR-materialer
- Hud: `roughnessFactor=0.5` (0.4–0.6, ikke plast)
- Kropp: `baseColorFactor` fra hudfarge, `metallicFactor=0`
- Eksporteres som glTF PBR metallic-roughness

### 3. Proporsjons-hooks (for fremtidig AI-fitting)
```python
from pipeline import get_body_scale_defaults, run_pipeline

scales = get_body_scale_defaults()  # heightScale, shoulderWidthScale, chestScale, waistScale, hipScale, legLengthScale
run_pipeline(..., body_scales={"shoulderWidthScale": 1.1, "legLengthScale": 1.05})
```

### 4. Viewer-lys (GLBViewer.js)
- **HemisphereLight:** Himmel (lys) + bakke (skygge)
- **DirectionalLight (key):** Styrke 1.0, castShadow, PCFSoftShadowMap
- **Fill + rim:** Bløte skygger
- **Shadow plane:** Mottar skygge under modellen
- **Tone mapping:** ACESFilmicToneMapping (hvis støttet)

### 5. Polybudsjett
- Holdt under ~50–80k trekanter
- Teksturer ≤ 2K (TEX_SIZE=1024)

## Verifisering
1. Start backend, kjør `python generate_default_avatar.py`
2. Åpne app, la backend kjøre – default-avatar vises
3. Legg til fjes + kropp, trykk «Generer 3D-avatar med AI»
4. Sjekk: rundet silhuett, hendere, føtter, skygge, ingen flat/grå plastikk-look
