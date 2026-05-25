COMMIT_FRAMES = 10
SPACE_FRAMES = 15
SENTENCE_END_FRAMES = 60
CONFIDENCE_THRESHOLD = 0.85


class SentenceBuffer:
    """
    Converts per-frame letter predictions into a running sentence.

    Interaction grammar:
      - Same letter held >= commit_frames consecutive frames → commit letter to current word
      - 'space' class held >= commit_frames frames → end current word, insert word space
      - 'del' class held >= commit_frames frames → backspace last character
      - No hand >= space_frames frames → insert word space
      - No hand >= sentence_end_frames frames → reset everything
      - Anti-repeat: after committing a letter, the same letter must be
        released (hand removed or different letter shown) before it can commit again.
        Prevents "HELLOOO" from a single held gesture.

    All frame-count parameters are constructor args so the demo can tune them
    without touching this file.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        commit_frames: int = COMMIT_FRAMES,
        space_frames: int = SPACE_FRAMES,
        sentence_end_frames: int = SENTENCE_END_FRAMES,
    ):
        self._confidence_threshold = confidence_threshold
        self._commit_frames = commit_frames
        self._space_frames = space_frames
        self._sentence_end_frames = sentence_end_frames

        self._sentence: str = ""        # completed words (with trailing spaces)
        self._current_word: str = ""    # word currently being spelled
        self._pending: str | None = None
        self._pending_count: int = 0
        self._no_hand_count: int = 0
        self._last_committed: str | None = None  # anti-repeat guard

    # ------------------------------------------------------------------
    def update(self, letter: str | None, confidence: float = 0.0) -> None:
        """Call once per frame. Pass letter=None when no hand is detected."""
        if letter is None:
            self._pending = None
            self._pending_count = 0
            self._last_committed = None  # hand removed → can re-sign same letter

            self._no_hand_count += 1
            if self._no_hand_count >= self._sentence_end_frames:
                self.reset()
            elif self._no_hand_count == self._space_frames:
                self._insert_space()
            return

        self._no_hand_count = 0

        if confidence < self._confidence_threshold:
            self._pending = None
            self._pending_count = 0
            return

        # Anti-repeat: last committed letter still showing → wait for release
        if letter == self._last_committed:
            return

        # Letter changed → clear guard, restart counter
        if letter != self._pending:
            self._last_committed = None
            self._pending = letter
            self._pending_count = 1
            return

        self._pending_count += 1
        if self._pending_count < self._commit_frames:
            return

        # --- commit ---
        self._last_committed = letter
        self._pending = None
        self._pending_count = 0

        if letter == "del":
            self._backspace()
        elif letter == "space":
            self._insert_space()
        else:
            self._current_word += letter

    # ------------------------------------------------------------------
    def get_state(self) -> dict:
        """
        Returns:
            {
                "pending_letter":   str | None,  # held but not yet committed
                "pending_progress": float,        # 0.0–1.0 for progress bar
                "current_word":     str,          # word being spelled right now
                "sentence":         str,          # full display string
                "is_no_hand":       bool,
            }
        """
        return {
            "pending_letter": self._pending,
            "pending_progress": (
                min(self._pending_count / self._commit_frames, 1.0)
                if self._pending is not None else 0.0
            ),
            "current_word": self._current_word,
            "sentence": self._sentence + self._current_word,
            "is_no_hand": self._no_hand_count > 0,
        }

    def reset(self) -> None:
        self._sentence = ""
        self._current_word = ""
        self._pending = None
        self._pending_count = 0
        self._no_hand_count = 0
        self._last_committed = None

    # ------------------------------------------------------------------
    def _insert_space(self) -> None:
        if self._current_word:
            self._sentence += self._current_word + " "
            self._current_word = ""
        elif self._sentence and not self._sentence.endswith(" "):
            self._sentence += " "

    def _backspace(self) -> None:
        if self._current_word:
            self._current_word = self._current_word[:-1]
        elif self._sentence:
            self._sentence = self._sentence[:-1]
