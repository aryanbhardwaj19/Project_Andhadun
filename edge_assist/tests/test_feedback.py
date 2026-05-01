"""
Integration test: fusion → haptic → speech pipeline.

Expected console + audio output:
  [TEST] Positive Feedback -> [HAPTIC] double-short   | audio: "bip. bip." then "Person recognized"
  [TEST] Negative Feedback -> [HAPTIC] long            | audio: "beeeeeeeep" then "Alert: Suspicious activity"
  [TEST] Neutral  Feedback -> [HAPTIC] single-short    | audio: "bip" then "Background scan active"
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from edge_assist.fusion import DecisionFusion, FeedbackEvent
from edge_assist.haptic import HapticSimulator
from edge_assist.audio import _AUDIO_QUEUE

# ── Lookup tables ──────────────────────────────────────────────────────────────
PATTERN_MAP = {
    FeedbackEvent.POSITIVE: 'double-short',
    FeedbackEvent.NEGATIVE: 'long',
    FeedbackEvent.NEUTRAL:  'single-short',
}

SPEECH_MAP = {
    FeedbackEvent.POSITIVE: "Person recognized",
    FeedbackEvent.NEGATIVE: "Alert: Suspicious activity",
    FeedbackEvent.NEUTRAL:  "Background scan active",
}

# Emotion sequence that should trigger each event after 5-frame majority vote
SCENARIOS = [
    ("Positive Feedback", ["happy"] * 5,   FeedbackEvent.POSITIVE),
    ("Negative Feedback", ["angry"] * 5,   FeedbackEvent.NEGATIVE),
    ("Neutral Feedback",  ["neutral"] * 5, FeedbackEvent.NEUTRAL),
]

def run_test():
    haptic = HapticSimulator()
    FACE_AREA = 0.10  # above proximity_threshold=0.05

    all_passed = True

    for label, emotions, expected_event in SCENARIOS:
        # Fresh fusion instance per scenario (avoids cooldown carry-over)
        fusion = DecisionFusion(buffer_size=5, cooldown_seconds=3.0, proximity_threshold=0.05)

        result = FeedbackEvent.NO_ALERT
        for emo in emotions:
            result = fusion.update(emo, FACE_AREA)

        print(f"\n[TEST] {label} -> ", end='', flush=True)

        # Assertion
        assert result == expected_event, (
            f"FAIL: expected {expected_event}, got {result}"
        )

        # Trigger haptic sim (queues beep sound)
        pattern = PATTERN_MAP[result]
        haptic.vibrate(pattern)

        # Queue follow-up speech on the SAME audio worker — ordering guaranteed
        haptic.audio.speak(SPEECH_MAP[result])

        # Wait for this scenario's audio to finish before moving to the next one
        _AUDIO_QUEUE.join()

    print("\n\n[TEST SUITE] All assertions passed.")
    # Wait for any remaining audio to finish
    _AUDIO_QUEUE.join()
    print("[TEST SUITE] Audio queue drained. Done.")

if __name__ == "__main__":
    run_test()