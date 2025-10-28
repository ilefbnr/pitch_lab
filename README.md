# Pitch Lab

Record, analyze, and perfect your business pitches with AI-powered insights. Includes:
- Recording and saving audio/video pitches
- Real-time camera/mic analysis
- Reports and live metrics

## Project structure

- `vocal-aid-pro/` — React + Vite frontend (Tabs: Record, Real-time, Video, My Pitches)
- `backend_Pitch/` — FastAPI backend (primary API used by the app)
- `models/` — Local ML models (Vosk, etc., gitignored)

## Requirements

- Windows 10/11
- Node.js 18+ and npm
- Python 3.10 or 3.11 (recommended for librosa compatibility)
- Git
- ffmpeg (for decoding WebM/OGG/Opus) — optional but recommended

## Backend setup (FastAPI)

1) Create and activate a virtual environment
```powershell
cd "C:\Users\ilefb\OneDrive\Bureau\pitch lab\backend_Pitch"
py -3.11 -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Install dependencies
```powershell
pip install fastapi "uvicorn[standard]" python-multipart
pip install numpy scipy soundfile numba librosa
pip install vosk openai-whisper
# Optional (only if needed/online): transformers torch spacy
```

3) Download Vosk model (English small) and extract
```powershell
cd "C:\Users\ilefb\OneDrive\Bureau\pitch lab\backend_Pitch\models"
Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile "vosk-small-en-us.zip"
Expand-Archive -Path .\vosk-small-en-us.zip -DestinationPath . -Force
# Ensure the folder exists: .\vosk-model-small-en-us-0.15
```

If your code allows, set an absolute model path (optional):
```powershell
$env:VOSK_MODEL_DIR = "C:\Users\ilefb\OneDrive\Bureau\pitch lab\backend_Pitch\models\vosk-model-small-en-us-0.15"
```

4) Run the API (port 8000)
```powershell
cd "C:\Users\ilefb\OneDrive\Bureau\pitch lab\backend_Pitch"
& .\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```




## Frontend setup (Vite)

1) Install and run
```powershell
cd "C:\Users\ilefb\OneDrive\Bureau\pitch lab\vocal-aid-pro"
npm install
# set API URL to backend
$env:VITE_API_URL="http://localhost:8000"
npm run dev
```

- Open http://localhost:5173

The Pitch Lab page shows:
- Header: “Pitch Lab” + tagline
- Tabs:
  - Record (default): record audio, name it, save, wait for analysis (POST /pitches)
  - Real-time: start camera/mic, live metrics and analysis
  - Video: upload/process videos (if implemented)
  - My Pitches: list previously saved pitches

## API overview (common endpoints)

- GET `/health` — service status
- POST `/pitches` — multipart form
  - fields: `title`, `description` (optional), `audio_file` OR `video_file`
- GET `/pitches/{id}` — pitch details
- GET `/pitches/{id}/report` — analysis report (if generated)
- POST `/audio/realtime-emotion` — realtime audio chunks
- POST `/video/realtime-emotion` — realtime video frames
- WebSocket (if enabled): `/realtime-pitch` or `/ws/realtime/{session_id}`

Adjust to your actual routes if different.

## Troubleshooting

- ERR_CONNECTION_REFUSED
  - Backend not running or port mismatch. Ensure frontend uses `VITE_API_URL=http://localhost:8000`.

- “Form data requires python-multipart”
  - `pip install python-multipart` in the correct venv.

- Vosk “Failed to create a model”
  - Re-extract the model folder, ensure path is correct and local (OneDrive: Always keep on this device).
  - Use absolute `VOSK_MODEL_DIR`.

- librosa on Python 3.13
  - Prefer Python 3.10/3.11, or guard imports in code.

- spaCy/HuggingFace downloads fail offline
  - Use env toggles above and guard imports.

- ffmpeg missing (pydub decode issues)
  - Install and add to PATH:
    - https://ffmpeg.org/download.html
    - Verify: `ffmpeg -version`

## Git: selectively add files

- Use .gitignore (already provided) to exclude: venv, node_modules, uploads, models, .env, etc.
- Add only what you want:
```powershell
cd "C:\Users\ilefb\OneDrive\Bureau\pitch lab"
git init
git checkout -b main
git add vocal-aid-pro backend_Pitch pitchlab_backend README.md .gitignore
# If you want to exclude subfolders explicitly:
git reset backend_Pitch/models backend_Pitch/uploads
git commit -m "Initial commit"
```
- Or add paths one by one:
```powershell
git add vocal-aid-pro/src backend_Pitch/main.py backend_Pitch/voice_analysis.py README.md
```
- To unstage something:
```powershell
git restore --staged <path>
```

Optional: Git LFS for large media
```powershell
git lfs install
git lfs track "*.wav" "*.webm" "*.mp4" "*.zip"
git add .gitattributes
```

## License

MIT (update as needed).

## Acknowledgments

- FastAPI, Vosk, Whisper, librosa, React/Vite, and related open-source projects.