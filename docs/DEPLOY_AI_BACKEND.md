# Deploy AI-backend

## 1. Lokalt (testing)

**Viktig:** Start backend fra `ai-backend`-mappen, ellers feiler import av `main`.

```bash
cd ai-backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

For mannequin-basert avatar: kjør én gang `python download_mannequin.py` (nedlaster base mesh).  
Deretter `python generate_default_avatar.py` for forhåndsvisning.

Appen bruker `http://localhost:8000` (iOS) eller `http://10.0.2.2:8000` (Android emulator).

## 2. Railway (gratis)

1. Gå til [railway.app](https://railway.app)
2. "New Project" → "Deploy from GitHub repo"
3. Velg ditt repo (eller "Empty project" og last opp `ai-backend`-mappen)
4. I prosjektet: Settings → Root Directory: `ai-backend` (hvis backend er i undermappe)
5. Settings → Build: `pip install -r requirements.txt`
6. Settings → Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Deploy → "Generate Domain" for å få URL (f.eks. `https://tryon-ai.up.railway.app`)

## 3. Oppdater appen

I `src/config.js` – for produksjon, endre:

```js
const AI_BACKEND_URL = 'https://din-railway-url.up.railway.app';
```

## 4. Bruk i appen

1. Legg til fjes (selfie eller galleri)
2. Trykk **"Generer avatar med AI"**
3. Vent – AI-en bruker MediaPipe til å plukke ut fjeset og lage avatar
4. Avatar vises – lagre til profil
