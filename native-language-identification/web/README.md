# Web App (React + Flask)

A futuristic, neon-styled interface for Native Language Identification using HuBERT/MFCC.

## Structure
- Backend (Flask): `web/backend/app.py`
- Frontend (Vite + React + Tailwind): `web/frontend`

## Prerequisites
- Python 3.12 with project deps installed (includes Flask)
- Node 18+ and npm (for frontend)

## Backend: Run
```bash
cd native-language-identification
python3 web/backend/app.py
# API at http://localhost:5000
```

Health check:
```bash
curl http://localhost:5000/api/health
```

## Frontend: Run
```bash
cd native-language-identification/web/frontend
npm install
npm run dev
# App at http://localhost:5173
```

Set a custom API base (optional): create `.env` in `web/frontend`:
```
VITE_API_BASE=http://localhost:5000
```

## Predict flow
1. Open the web app
2. Upload WAV/MP3
3. Click Predict
4. See detected language, cuisines, and confidence bars

Notes:
- If no trained checkpoint is found under `models/checkpoints/<experiment>/best_model.pt`, backend will still run in demo mode (randomish outputs). Train for meaningful results.
- Uploaded files are saved to `data/uploads/` and a JSON of predictions is produced under `outputs/predictions/`.