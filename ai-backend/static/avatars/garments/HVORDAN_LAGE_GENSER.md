# Slik lager du en genser til avataren

Genseren må være en **GLB-fil** med samme **skjelett/rig** som den kvinnelige base-avataren (samme bonenavn). Du har tre hovedveier:

---

## 1. Blender (anbefalt)

### Steg A: Finn bonenavnene i avataren din

1. Åpne **Blender** (gratis: blender.org).
2. **File → Import → glTF 2.0 (.glb/.gltf)** og velg base-avataren din  
   (f.eks. `ai-backend/static/avatars/base_avatar.glb` eller `base_human_female.glb`).
3. I **Outliner** (høyre panel) finner du en **Armature** (ikon som skjelett).
4. Klikk på pil ved siden av Armature og utvid – da ser du alle **bonenavn** (f.eks. `mixamorig:Hips`, `mixamorig:Spine`, `mixamorig:LeftArm` osv.).  
   Skriv ned eller ta skjermbilde – genseren må bruke **nøyaktig samme navn**.

### Steg B: Lag eller importer genser-mesh

- **Enkel variant:** Lag en enkel form rundt overkropp:
  - **Add → Mesh → Cylinder** (eller **Cube**).
  - Juster med **Edit mode** (Tab): skaler, flytt og form den som en genser over torso og øvre armer.
- **Mer realistisk:** Last ned en ferdig genser-modell (f.eks. fra [Sketchfab](https://sketchfab.com) med CC-lisens, eller fra [Mixamo](https://www.mixamo.com) som allerede er rigget).

### Steg C: Rig genseren til avatarens skjelett

1. **Importer base-avataren** i samme scene (eller behold den du allerede importerte).
2. Velg **genser-meshen**, deretter **Shift** og velg **avatarens Armature** (ikke mesh).
3. **Ctrl+P → With Automatic Weights** (eller **With Empty Groups** og paint weights selv).
4. Sjekk at genseren nå har **samme Armature** som avataren (samme bonenavn).
5. **Fjern** avatarens synlige mesh fra eksport om du bare vil eksportere genseren:
   - Velg avatarens body-mesh(es) og skjul dem (H) eller flytt til annen layer.
6. **File → Export → glTF 2.0 (.glb/.gltf)**:
   - Velg **mappen**: `ai-backend/static/avatars/garments/`
   - Filnavn: **sweater_female.glb**
   - Kryss av: **Export skinning** (viktig).
   - Eksporter.

Da har du en genser som bruker samme rig som avataren og vil bevege seg med kroppen i appen.

---

## 2. MakeHuman (hvis avataren kommer derfra)

Hvis base-avataren din er laget eller eksportert fra **MakeHuman**:

1. Åpne MakeHuman og lag/en **female** figur.
2. Gå til **Geometry → Clothes** og velg eller last inn en genser/t-shirt.
3. **File → Export** og velg **glTF 2.0 (.glb)**.
4. I eksport-dialogen: eksporter **med skjelett** og **med klær**.  
   Du kan evt. eksportere kun klær-meshen og deretter i Blender knytte den til samme rig som base-avataren (som i Steg C over).

---

## 3. Enkel test uten skinning (uten rig)

For å bare **teste at lasting fungerer** kan du bruke en genser **uten** skinning:

1. I Blender: **Add → Mesh → UV Sphere** eller **Cylinder**, plasser rundt torso.
2. **Ikke** knytt den til noen armature.
3. Eksporter som **sweater_female.glb** til `ai-backend/static/avatars/garments/`.

I appen vil genseren da lastes og skaleres til avataren, men **beveger seg ikke** med kroppen – den ligger statisk over. Nyttig for å sjekke at filsti og lasting er riktig.

---

## Oppsummert

| Krav | Beskrivelse |
|------|-------------|
| **Fil** | `sweater_female.glb` |
| **Plassering** | `ai-backend/static/avatars/garments/sweater_female.glb` |
| **Skjelett** | Samme bonenavn og hierarki som female base-avatar (f.eks. base_avatar.glb / base_human_female.glb). |
| **Format** | glTF 2.0 (.glb), med skinning hvis du vil at den skal følge bevegelser. |

Etter at filen ligger på plass, start backend på nytt om nødvendig og bruk «Prøv genser» i appen.
