"""
Tests for SentenceBuffer. No webcam or model required — all inputs are mocked.
Run with: python test_buffer.py
"""

from buffer import SentenceBuffer


def _sign(buf: SentenceBuffer, letter: str, n: int, conf: float = 0.95) -> None:
    """Feed n frames of the same letter."""
    for _ in range(n):
        buf.update(letter, conf)


def _no_hand(buf: SentenceBuffer, n: int) -> None:
    """Feed n frames with no hand detected."""
    for _ in range(n):
        buf.update(None)


# ---------------------------------------------------------------------------

def test_spell_hi():
    """Spelling HI produces current_word == 'HI'."""
    buf = SentenceBuffer(commit_frames=3, space_frames=10, sentence_end_frames=30)
    _sign(buf, "H", 3)
    _no_hand(buf, 1)     # release so anti-repeat clears
    _sign(buf, "I", 3)
    state = buf.get_state()
    assert state["current_word"] == "HI", f"expected 'HI', got {state['current_word']!r}"
    assert state["sentence"] == "HI", state["sentence"]
    print("PASS  test_spell_hi")


def test_spell_hi_world_with_pause():
    """Spelling HI, pausing, then WORLD produces 'HI WORLD' in sentence."""
    buf = SentenceBuffer(commit_frames=3, space_frames=5, sentence_end_frames=30)

    _sign(buf, "H", 3); _no_hand(buf, 1)
    _sign(buf, "I", 3); _no_hand(buf, 1)

    # Pause long enough to insert a word space (5 frames)
    _no_hand(buf, 5)

    _sign(buf, "W", 3); _no_hand(buf, 1)
    _sign(buf, "O", 3); _no_hand(buf, 1)
    _sign(buf, "R", 3); _no_hand(buf, 1)
    _sign(buf, "L", 3); _no_hand(buf, 1)
    _sign(buf, "D", 3)

    state = buf.get_state()
    assert state["sentence"] == "HI WORLD", f"got {state['sentence']!r}"
    print("PASS  test_spell_hi_world_with_pause")


def test_anti_repeat():
    """Holding a letter for 30 frames after commit should only produce one 'H'."""
    buf = SentenceBuffer(commit_frames=3, space_frames=10, sentence_end_frames=60)

    # Commit H, then keep signing H for 30 more frames without releasing
    _sign(buf, "H", 30)

    state = buf.get_state()
    assert state["current_word"] == "H", (
        f"anti-repeat failed — expected 'H', got {state['current_word']!r}"
    )
    print("PASS  test_anti_repeat")


def test_anti_repeat_allows_resign():
    """After releasing (no-hand), the same letter can commit again."""
    buf = SentenceBuffer(commit_frames=3, space_frames=10, sentence_end_frames=60)

    _sign(buf, "A", 3)   # commits A
    _no_hand(buf, 1)      # release — clears anti-repeat
    _sign(buf, "A", 3)   # should commit a second A

    state = buf.get_state()
    assert state["current_word"] == "AA", f"got {state['current_word']!r}"
    print("PASS  test_anti_repeat_allows_resign")


def test_low_confidence_ignored():
    """Predictions below the confidence threshold must not accumulate."""
    buf = SentenceBuffer(commit_frames=3, confidence_threshold=0.85)

    for _ in range(10):
        buf.update("Z", 0.5)   # below threshold

    state = buf.get_state()
    assert state["pending_letter"] is None, "low-conf letter should not be pending"
    assert state["current_word"] == "", state["current_word"]
    print("PASS  test_low_confidence_ignored")


def test_backspace():
    """del class removes the last committed character."""
    buf = SentenceBuffer(commit_frames=3, space_frames=10, sentence_end_frames=60)

    _sign(buf, "H", 3); _no_hand(buf, 1)
    _sign(buf, "I", 3); _no_hand(buf, 1)
    _sign(buf, "del", 3)  # removes I

    state = buf.get_state()
    assert state["current_word"] == "H", f"got {state['current_word']!r}"
    print("PASS  test_backspace")


def test_sentence_end_resets():
    """Long no-hand period resets the buffer entirely."""
    buf = SentenceBuffer(commit_frames=3, space_frames=5, sentence_end_frames=10)

    _sign(buf, "H", 3); _no_hand(buf, 1)
    _sign(buf, "I", 3)
    _no_hand(buf, 10)   # triggers sentence end

    state = buf.get_state()
    assert state["sentence"] == "", f"got {state['sentence']!r}"
    assert state["current_word"] == "", state["current_word"]
    print("PASS  test_sentence_end_resets")


def test_get_state_shape():
    """get_state() returns all expected keys with correct types."""
    buf = SentenceBuffer()
    state = buf.get_state()
    assert set(state.keys()) == {
        "pending_letter", "pending_progress", "current_word", "sentence", "is_no_hand"
    }
    assert state["pending_letter"] is None
    assert state["pending_progress"] == 0.0
    assert state["current_word"] == ""
    assert state["sentence"] == ""
    assert state["is_no_hand"] is False
    print("PASS  test_get_state_shape")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_get_state_shape()
    test_spell_hi()
    test_spell_hi_world_with_pause()
    test_anti_repeat()
    test_anti_repeat_allows_resign()
    test_low_confidence_ignored()
    test_backspace()
    test_sentence_end_resets()
    print("\nAll tests passed.")
