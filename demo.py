import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from buffer import SentenceBuffer
from infer_mlp import predict

ROOT = Path(__file__).parent
TASK_PATH = ROOT / "models" / "hand_landmarker.task"


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


def _draw_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_overlay(
    frame: np.ndarray,
    pending: str | None,
    confidence: float,
    progress: float,
    sentence: str,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # --- top bar: sentence ---
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 64), (0, 0, 0), -1)
    frame = cv2.addWeighted(bar, 0.55, frame, 0.45, 0)
    display = sentence[-48:] if len(sentence) > 48 else sentence
    cv2.putText(frame, display or "(start signing)",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # --- bottom panel: letter + confidence + progress bar ---
    if pending:
        # big letter
        cv2.putText(frame, pending,
                    (w // 2 - 28, h - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 255, 120), 5, cv2.LINE_AA)
        # confidence label
        cv2.putText(frame, f"{confidence:.0%}",
                    (w // 2 - 22, h - 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (210, 210, 210), 2, cv2.LINE_AA)
        # commit-progress bar
        bw, bh = 200, 14
        bx, by = w // 2 - bw // 2, h - 50
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * progress), by + bh), (0, 220, 100), -1)

    # --- controls hint ---
    cv2.putText(frame, "q = quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


def run(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")

    buf = SentenceBuffer()
    t0 = time.time()
    last_confidence = 0.0

    with _make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so it acts like a selfie camera
            frame = cv2.flip(frame, 1)
            timestamp_ms = int((time.time() - t0) * 1000)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            if result.hand_landmarks:
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                    dtype=np.float32,
                )
                letter, conf = predict(coords)
                last_confidence = conf
                buf.update(letter, conf)
                _draw_landmarks(frame, result.hand_landmarks[0])
            else:
                last_confidence = 0.0
                buf.update(None)

            state = buf.get_state()
            frame = _draw_overlay(
                frame,
                state["pending_letter"],
                last_confidence,
                state["pending_progress"],
                state["sentence"],
            )

            cv2.imshow("ASL Demo — press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Final sentence:", buf.get_state()["sentence"])


if __name__ == "__main__":
    run()
