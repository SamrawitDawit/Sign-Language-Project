COMMIT_FRAMES = 15        # frames same letter must be held to commit
SPACE_FRAMES = 20         # no-hand frames before inserting a word space
CONFIDENCE_THRESHOLD = 0.85


class SentenceBuffer:
    """
    Converts per-frame letter predictions into a running sentence.

    Interaction grammar:
      - Same letter held >= COMMIT_FRAMES consecutive frames → commit letter
      - 'space' class held >= COMMIT_FRAMES frames → commit a word space
      - 'del' class held >= COMMIT_FRAMES frames → delete last committed char
      - No hand detected for >= SPACE_FRAMES frames → insert word space
      - Prediction confidence below threshold → treated as no prediction
    """

    def __init__(
        self,
        commit_frames: int = COMMIT_FRAMES,
        space_frames: int = SPACE_FRAMES,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        self._commit_frames = commit_frames
        self._space_frames = space_frames
        self._confidence_threshold = confidence_threshold

        self._sentence: str = ""
        self._pending: str | None = None
        self._pending_count: int = 0
        self._no_hand_count: int = 0

    def update(self, letter: str | None, confidence: float = 0.0) -> None:
        """Call once per frame. Pass letter=None when no hand is detected."""
        if letter is None:
            self._pending = None
            self._pending_count = 0
            self._no_hand_count += 1
            if self._no_hand_count >= self._space_frames:
                if self._sentence and not self._sentence.endswith(" "):
                    self._sentence += " "
                self._no_hand_count = 0
            return

        self._no_hand_count = 0

        if confidence < self._confidence_threshold:
            self._pending = None
            self._pending_count = 0
            return

        if letter != self._pending:
            self._pending = letter
            self._pending_count = 1
            return

        self._pending_count += 1
        if self._pending_count < self._commit_frames:
            return

        # --- commit ---
        if letter == "del":
            self._sentence = self._sentence[:-1]
        elif letter == "space":
            if not self._sentence.endswith(" "):
                self._sentence += " "
        else:
            self._sentence += letter

        self._pending = None
        self._pending_count = 0

    def get_current_sentence(self) -> str:
        return self._sentence

    def get_pending_letter(self) -> str | None:
        """Letter currently being held but not yet committed."""
        return self._pending

    def get_progress(self) -> float:
        """Fraction of commit threshold reached for the pending letter (0.0–1.0)."""
        if self._pending is None:
            return 0.0
        return min(self._pending_count / self._commit_frames, 1.0)

    def clear(self) -> None:
        self._sentence = ""
        self._pending = None
        self._pending_count = 0
        self._no_hand_count = 0


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    buf = SentenceBuffer(commit_frames=3, space_frames=4, confidence_threshold=0.5)

    # Hold 'H' for 3 frames → commits
    for _ in range(3):
        buf.update("H", 0.95)
    assert buf.get_current_sentence() == "H", buf.get_current_sentence()

    # Hold 'I' for 3 frames → commits
    for _ in range(3):
        buf.update("I", 0.95)
    assert buf.get_current_sentence() == "HI"

    # No hand for 4 frames → space
    for _ in range(4):
        buf.update(None)
    assert buf.get_current_sentence() == "HI "

    # Hold 'del' for 3 frames → removes space
    for _ in range(3):
        buf.update("del", 0.95)
    assert buf.get_current_sentence() == "HI"

    # Low-confidence prediction → ignored
    buf.update("Z", 0.3)
    assert buf.get_pending_letter() is None

    print("All assertions passed.")
    print("Sentence:", repr(buf.get_current_sentence()))