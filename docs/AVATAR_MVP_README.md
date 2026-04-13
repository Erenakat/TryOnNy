# Avatar MVP – End-to-End Setup

## Overview

When the user uploads **face** and **body front** (and optionally **body side**) and taps **"Generer 3D-avatar med AI"**:

1. Images are uploaded as multipart/form-data (**do not** set `Content-Type` manually in React Native).
2. Backend (Python) creates a job and returns `{ jobId }`.
3. App polls `GET /avatar/jobs/:jobId` every 2 s (max 3 min).
4. When `status: "done"`, `avatarUrl` is the path to the GLB (app prepends `AI_BACKEND_URL`).
5. App renders the 3D avatar in a GLB viewer with rotation (0–360° slider + buttons).
6. User can **Lagre avatar** to persist `avatarUrl` and rotation on profile (AsyncStorage).

## Quick Start

### 1. Start AI Backend (Python)

```bash
cd ai-backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Backend runs at `http://localhost:8000`.

### 2. Start Mobile App

```bash
cd ..
npm install
npx expo start
```

- **iOS Simulator**: `AI_BACKEND_URL` = `http://localhost:8000`
- **Android Emulator**: use `http://10.0.2.2:8000` (set in `src/config.js` for `__DEV__`)
- **Physical device**: set `AI_BACKEND_URL` to your machine’s LAN IP (e.g. `http://192.168.1.100:8000`)

### 3. Test Flow

1. Add **face** image (Fjes → camera or gallery).
2. Add **body front** image (Kropp foran → camera or gallery).
3. (Optional) Add **body side** (Kropp side).
4. Tap **"Generer 3D-avatar med AI"**.
5. Wait for progress (percentage + step message: "Analyzing body shape", "Reconstructing face", etc.).
6. 3D avatar appears; use slider, arrows, or drag on the viewer to rotate (0–360°).
7. Tap **"Lagre avatar"** to persist.

## API Spec

### POST /avatar/jobs

- **Content-Type**: `multipart/form-data` (do **not** set manually in RN).
- **Fields**: `face` (image), `bodyFront` (image), `bodySide` (image, optional).
- **Response**: `{ jobId: string }`

### GET /avatar/jobs/:jobId

- **Response**: `{ status, progress?, progress_message?, avatarUrl?, error? }`  
  - `status`: `"queued"` | `"processing"` | `"done"` | `"failed"`  
  - `progress_message`: step description (e.g. "Analyzing body shape", "Reconstructing face") – app shows this under the percentage  
  - `avatarUrl`: e.g. `"/static/avatars/<jobId>.glb"` (client prepends base URL)

### Static Files

- Avatars: `GET /static/avatars/:id.glb`

## Test Plan & Debugging

To avoid “nothing happens”, follow this and check logs.

### Backend

1. **Health**: `curl http://localhost:8000/health` → `{"ok":true}`.
2. **Create job** (replace paths with real files):
   ```bash
   curl -X POST http://localhost:8000/avatar/jobs \
     -F "face=@/path/to/face.jpg" \
     -F "bodyFront=@/path/to/body.jpg"
   ```
   → Expect `{"jobId":"..."}`.
3. **Poll**: `curl http://localhost:8000/avatar/jobs/<jobId>`  
   → Expect `status` to go `queued` → `processing` → `done` and `avatarUrl` set.
4. **Logs**: Backend logs `job <id> progress ...` and `job <id> done -> /static/avatars/...`. If the job fails, check traceback and `error` in GET response.

### App

1. **Config**: In `src/config.js`, `AI_BACKEND_URL` must match where the backend runs (see Quick Start).
2. **Upload**: Do **not** set `Content-Type` on the fetch that sends `FormData`; let the runtime set it (with boundary).
3. **Polling**: App polls every 2 s; after 3 min it shows “Tidsavbrudd”. Check that backend is not returning 404/500 for the job.
4. **GLB URL**: App uses `avatarUrl` as full URL: if backend returns `"/static/avatars/x.glb"`, app uses `AI_BACKEND_URL + avatarUrl`.
5. **Console**: Use `console.warn('create job error', e)` and `console.warn('poll error', e)` to see network or JSON errors.

### Happy Path Checklist

- [ ] Backend running; `/health` returns 200.
- [ ] Face + body front selected in app.
- [ ] Tap “Generer 3D-avatar med AI” → progress text and % appear.
- [ ] After 10–60 s, 3D avatar appears (GLB viewer).
- [ ] Slider and arrows change rotation (0–360°).
- [ ] “Lagre avatar” saves; after restart, avatar loads from profile (`avatarUrl` persisted).

## Common Failures

| Symptom | Cause | Fix |
|--------|--------|-----|
| “Kunne ikke nå server” | Backend not running / wrong URL | Start ai-backend; check `AI_BACKEND_URL` in `src/config.js` |
| Poll timeout / “Tidsavbrudd” | Job failed or backend slow | Check backend logs; ensure pipeline (trimesh, mediapipe) runs |
| Images not uploading | FormData Content-Type set manually | Remove any `Content-Type` header on fetch |
| Blank 3D viewer | CORS or wrong avatarUrl | Ensure full URL; allow CORS on backend |
| Android “Network request failed” | Cleartext HTTP blocked | `usesCleartextTraffic: true` in app config if using HTTP |
