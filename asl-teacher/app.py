"""
ASL Sign Language Teacher — FastAPI backend.

Run:
    pip install fastapi uvicorn[standard]
    python teacher/app.py

Then open http://localhost:8000 in your browser.

Reference images: drop 24 JPGs into teacher/static/asl/
    A.jpg  B.jpg  C.jpg  …  Y.jpg   (no J, no Z)
"""
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ── Config ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # project root
TASK_PATH = ROOT / "models" / "hand_landmarker.task"

# Letters to teach — J and Z require motion, so they are excluded
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Frames the user must hold the correct sign before advancing (~1 s at 12 fps)
COMMIT_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.85

# ── Import project modules ──────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from infer_mlp import predict  # noqa: E402

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/letters")
async def get_letters():
    return {"letters": LETTERS}


# ── MediaPipe helper ────────────────────────────────────────────────────────
def _make_landmarker() -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


# ── WebSocket ───────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    t0 = time.time()
    correct_count = 0
    current_letter_idx = 0

    try:
        with _make_landmarker() as landmarker:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)

                # Client can jump to a specific letter (e.g. after manual skip)
                if msg.get("type") == "set_letter":
                    current_letter_idx = int(msg["index"]) % len(LETTERS)
                    correct_count = 0
                    continue

                frame_b64 = msg.get("frame", "")
                if not frame_b64:
                    continue

                # Decode JPEG frame sent from the browser
                img_bytes = base64.b64decode(frame_b64)
                arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                timestamp_ms = int((time.time() - t0) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, timestamp_ms)

                target = LETTERS[current_letter_idx]
                response: dict = {
                    "target":        target,
                    "letter_idx":    current_letter_idx,
                    "total_letters": len(LETTERS),
                    "detected":      None,
                    "confidence":    0.0,
                    "correct":       False,
                    "progress":      0.0,
                    "advance":       False,
                    "landmarks":     [],
                }

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
                    )
                    letter, conf = predict(coords)
                    detected = letter.upper()
                    is_correct = detected == target and conf >= CONFIDENCE_THRESHOLD

                    response["detected"]   = detected
                    response["confidence"] = round(conf, 3)
                    response["landmarks"]  = [{"x": lm.x, "y": lm.y} for lm in lms]
                    response["correct"]    = is_correct

                    if is_correct:
                        correct_count += 1
                    else:
                        correct_count = max(0, correct_count - 1)
                else:
                    # No hand — decay progress
                    correct_count = max(0, correct_count - 2)

                response["progress"] = min(correct_count / COMMIT_FRAMES, 1.0)

                if correct_count >= COMMIT_FRAMES:
                    response["advance"] = True
                    correct_count = 0
                    current_letter_idx = (current_letter_idx + 1) % len(LETTERS)
                    response["letter_idx"] = current_letter_idx

                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
