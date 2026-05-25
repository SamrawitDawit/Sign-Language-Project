"""
ASL Sign Language Teacher — FastAPI backend with timer, stats & retraining.

Run:
    uv run uvicorn asl-teacher.app:app --host 0.0.0.0 --port 8000
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

ROOT = Path(__file__).parent.parent
TASK_PATH = ROOT / "models" / "hand_landmarker.task"

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

COMMIT_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.85
TIMER_SECONDS = 10
MAX_RETRAIN_LETTERS = 3

sys.path.insert(0, str(ROOT))
from infer_mlp import predict  # noqa: E402

app = FastAPI()
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/letters")
async def get_letters():
    return {"letters": LETTERS}


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


def _init_stats():
    return {"attempts": 0, "wrong": 0, "committed": False, "timed_out": False, "skipped": False}


def _get_retrain_targets(stats: dict) -> list[str]:
    scored = [
        (letter, s["wrong"])
        for letter, s in stats.items()
        if not s["committed"] or s["wrong"] > 0
    ]
    scored.sort(key=lambda x: -x[1])
    return [letter for letter, _ in scored[:MAX_RETRAIN_LETTERS]]


# ── WebSocket ───────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    t0 = time.time()

    phase = "main"
    timer_mode = False
    correct_count = 0
    letter_start_time = time.time()
    current_letter_idx = 0
    retrain_queue: list[str] = []
    stats: dict[str, dict] = {l: _init_stats() for l in LETTERS}

    def _reset_letter():
        nonlocal correct_count, letter_start_time
        correct_count = 0
        letter_start_time = time.time()

    def _advance():
        nonlocal current_letter_idx, correct_count, phase, retrain_queue, letter_start_time
        correct_count = 0
        letter_start_time = time.time()

        if phase == "main":
            current_letter_idx = (current_letter_idx + 1) % len(LETTERS)
            if current_letter_idx == 0:
                targets = _get_retrain_targets(stats)
                if targets:
                    phase = "retrain"
                    retrain_queue = list(targets)
                    current_letter_idx = LETTERS.index(retrain_queue[0])
                else:
                    current_letter_idx = -1  # signal done
        else:
            if retrain_queue:
                retrain_queue.pop(0)
            if retrain_queue:
                current_letter_idx = LETTERS.index(retrain_queue[0])
            else:
                current_letter_idx = -1  # signal done

    try:
        with _make_landmarker() as landmarker:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "frame")

                if current_letter_idx == -1:
                    target = LETTERS[0]
                else:
                    target = LETTERS[current_letter_idx]

                # ── Control messages ────────────────────────────────────────
                if msg_type == "set_letter":
                    current_letter_idx = int(msg["index"]) % len(LETTERS)
                    _reset_letter()
                    continue

                if msg_type == "toggle_timer":
                    timer_mode = not timer_mode
                    _reset_letter()
                    continue

                if msg_type == "skip":
                    if current_letter_idx >= 0:
                        stats[target]["skipped"] = True
                        _advance()
                        # Send immediate confirmation so UI updates without waiting for next frame
                        skip_response = {
                            "type": "skip_confirm",
                            "letter_idx": current_letter_idx,
                            "target": LETTERS[current_letter_idx] if current_letter_idx >= 0 else "",
                            "done": current_letter_idx == -1,
                            "advance": True,
                            "advance_reason": "skip",
                        }
                        await websocket.send_text(json.dumps(skip_response))
                    continue

                if msg_type == "restart":
                    phase = "main"
                    retrain_queue = []
                    timer_mode = False
                    stats = {l: _init_stats() for l in LETTERS}
                    current_letter_idx = 0
                    _reset_letter()
                    continue

                # ── Frame processing ────────────────────────────────────────
                frame_b64 = msg.get("frame", "")
                if not frame_b64:
                    continue

                img_bytes = base64.b64decode(frame_b64)
                arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                timestamp_ms = int((time.time() - t0) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, timestamp_ms)

                elapsed = time.time() - letter_start_time
                time_remaining = max(0, TIMER_SECONDS - elapsed) if timer_mode else None

                done = current_letter_idx == -1

                response: dict = {
                    "target":         target,
                    "letter_idx":     current_letter_idx,
                    "total_letters":  len(LETTERS),
                    "detected":       None,
                    "confidence":     0.0,
                    "correct":        False,
                    "progress":       0.0,
                    "advance":        False,
                    "advance_reason": None,
                    "landmarks":      [],
                    "timer_mode":     timer_mode,
                    "time_remaining": time_remaining,
                    "letter_stats":   dict(stats.get(target, _init_stats())),
                    "phase":          phase,
                    "retrain_queue":  retrain_queue,
                    "done":           done,
                }

                if done:
                    await websocket.send_text(json.dumps(response))
                    continue

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    coords = np.array(
                        [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
                    )
                    letter, conf = predict(coords)
                    detected = letter.upper()
                    is_correct = detected == target and conf >= CONFIDENCE_THRESHOLD

                    response["detected"] = detected
                    response["confidence"] = round(conf, 3)
                    response["landmarks"] = [{"x": lm.x, "y": lm.y} for lm in lms]
                    response["correct"] = is_correct

                    if detected:
                        stats[target]["attempts"] += 1
                    if detected and detected != target:
                        stats[target]["wrong"] += 1

                    if is_correct:
                        correct_count += 1
                    else:
                        correct_count = max(0, correct_count - 1)
                else:
                    correct_count = max(0, correct_count - 2)

                response["progress"] = min(correct_count / COMMIT_FRAMES, 1.0)

                # ── Check for advance ─────────────────────────────────────
                advance = False
                advance_reason = None

                if correct_count >= COMMIT_FRAMES:
                    advance = True
                    advance_reason = "commit"
                    stats[target]["committed"] = True
                elif timer_mode and elapsed >= TIMER_SECONDS:
                    advance = True
                    advance_reason = "timed_out"
                    stats[target]["timed_out"] = True

                if advance:
                    response["advance"] = True
                    response["advance_reason"] = advance_reason
                    response["letter_stats"] = dict(stats.get(target, _init_stats()))
                    _advance()
                    response["letter_idx"] = current_letter_idx

                    if current_letter_idx == -1:
                        response["done"] = True
                        # Include retrain queue in final response
                        if phase == "retrain":
                            response["final_stats"] = {
                                "phase": phase,
                                "retrain_letters": retrain_queue,
                            }

                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        pass
