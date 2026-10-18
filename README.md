# AURA — Adaptive Understanding & Reasoning Architecture

Real-time visual intelligence system: YOLO object detection, multi-object tracking, OCR, gesture control, human activity recognition (HAR), voice assistant, RAG memory, and remote mobile camera streaming — all in a live web dashboard.

---

## Quick Start (First Time Setup)

### Prerequisites
- **Python 3.10+** — [python.org](https://www.python.org/downloads/)
- **Node.js 18+** — [nodejs.org](https://nodejs.org/) (for the web dashboard)
- **Git** — [git-scm.com](https://git-scm.com/)

### 1. Clone the repository
```bash
git clone https://github.com/Git-Nova-Coders/AURA.git
cd AURA
git checkout dev
```

### 2. Run the setup script (Windows)
```batch
setup.bat
```
This automatically:
- Creates a Python virtual environment (`.venv`)
- Installs all dependencies from `requirements.txt`
- Detects your GPU and installs the right PyTorch version (CUDA or CPU)
- Builds the React web frontend

### 3. Start AURA
```bash
# Activate the virtual environment first
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/macOS

# Run the dashboard
python run_dashboard.py
```

The browser opens automatically at **http://localhost:8420**

---

## Running Options

```bash
# Default (webcam + all features)
python run_dashboard.py

# Webcam with custom model
python run_dashboard.py --source 0 --model yolov8m-worldv2.pt

# Video file
python run_dashboard.py --source video.mp4

# Synthetic demo (no camera needed)
python run_dashboard.py --source synthetic

# Enable HTTPS for mobile camera streaming
python run_dashboard.py --ssl

# All options
python run_dashboard.py --help
```

---

## Remote Camera (Use Phone as Camera)

Stream your phone's camera feed to the AURA laptop dashboard with full object detection.

**Step 1** — Start AURA with SSL:
```bash
python run_dashboard.py --ssl
```

**Step 2** — The terminal will show a URL like:
```
✓ Remote Camera  →  https://10.186.149.246:8420/remote-camera
```

**Step 3** — Open that URL on your phone.
- You'll see a "connection not private" warning — tap **Advanced → Proceed** (one-time only, because of the self-signed cert)
- Tap **Allow** for camera permission
- Tap **START STREAMING** (or it auto-starts)

**Step 4** — Watch the live feed with detections on your laptop screen!

---

## Updating (After Team Push)

```bash
git checkout dev
git pull origin dev

# If new packages were added:
.venv\Scripts\activate
pip install -r requirements.txt

# Rebuild frontend if web files changed:
cd web && npm install && npm run build && cd ..

python run_dashboard.py
```

---

## Project Structure

```
AURA/
├── run_dashboard.py     # Main launcher
├── requirements.txt     # Python dependencies
├── setup.bat            # One-click Windows setup
├── interface/           # FastAPI server, WebSocket, bridge
├── vision/              # Camera, YOLO detector, tracker, features
├── ml/har/              # Human Activity Recognition (HAR) models
├── ocr/                 # EasyOCR text extraction
├── brain/               # Conversation engine, RAG, memory
├── ann/                 # Reliability ANN inference
├── voice/               # STT/TTS voice assistant
├── knowledge/           # Knowledge base and retriever
├── scripts/             # Utility scripts (SSL cert generator, etc.)
├── certs/               # Auto-generated SSL certs (gitignored)
├── web/                 # React dashboard frontend
│   ├── src/
│   └── dist/            # Built frontend (gitignored, auto-built)
├── data/                # HAR training data
├── models/              # Trained model weights
└── tests/               # Unit and integration tests
```

---

## Troubleshooting

### "Module not found" error
Make sure the virtual environment is activated:
```bash
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/macOS
```
Then reinstall:
```bash
pip install -r requirements.txt
```

### Dashboard doesn't open / blank page
Rebuild the frontend:
```bash
cd web && npm install && npm run build && cd ..
```

### Phone camera blocked
Use the `--ssl` flag when starting AURA:
```bash
python run_dashboard.py --ssl
```
Then accept the certificate warning on your phone (one-time).

### Model file not found
Download the default model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m-worldv2.pt')"
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object Detection | YOLOv8 / YOLO-World (Ultralytics) |
| Pose & Gesture | MediaPipe |
| Activity Recognition | CNN-LSTM (PyTorch) |
| OCR | EasyOCR |
| Tracking | Custom IoU + EMA tracker |
| Web Backend | FastAPI + WebSocket |
| Web Frontend | React + Vite |
| Voice | SoundDevice + STT/TTS |
| Knowledge | TF-IDF RAG + Wikipedia |
