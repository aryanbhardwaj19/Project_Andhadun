import sys
import os
import time
import unittest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusion import DecisionFusion, FeedbackEvent

class TestFusion(unittest.TestCase):
    def setUp(self):
        self.fusion = DecisionFusion(buffer_size=5, cooldown_seconds=1.0, proximity_threshold=0.1)

    def test_proximity_gate(self):
        # Even with 5 consistent emotions, if area is too small, no alert
        for _ in range(5):
            res = self.fusion.update("happy", 0.05)
        self.assertEqual(res, FeedbackEvent.NO_ALERT)

    def test_majority_vote_positive(self):
        # 3 positive, 2 negative = POSITIVE
        self.fusion.update("happy", 0.2)
        self.fusion.update("happy", 0.2)
        self.fusion.update("angry", 0.2)
        self.fusion.update("angry", 0.2)
        res = self.fusion.update("happy", 0.2)
        self.assertEqual(res, FeedbackEvent.POSITIVE)

    def test_majority_vote_negative(self):
        # 3 angry = NEGATIVE
        self.fusion.update("angry", 0.2)
        self.fusion.update("angry", 0.2)
        self.fusion.update("happy", 0.2)
        self.fusion.update("happy", 0.2)
        res = self.fusion.update("angry", 0.2)
        self.assertEqual(res, FeedbackEvent.NEGATIVE)

    def test_cooldown(self):
        # Trigger first alert
        for _ in range(5):
            res = self.fusion.update("happy", 0.2)
        self.assertEqual(res, FeedbackEvent.POSITIVE)
        
        # Immediate next frame (within 1s) should be NO_ALERT despite consistency
        res = self.fusion.update("happy", 0.2)
        self.assertEqual(res, FeedbackEvent.NO_ALERT)
        
        # Wait for cooldown
        time.sleep(1.1)
        # One more frame to push buffer or just fill it? 
        # Since it's a deque of 5, the last 5 are still "happy"
        res = self.fusion.update("happy", 0.2)
        self.assertEqual(res, FeedbackEvent.POSITIVE)

    def test_buffer_reset_on_proximity_fail(self):
        # Fill 4 frames with happy
        for _ in range(4):
            self.fusion.update("happy", 0.2)
        # 5th frame fails proximity
        res = self.fusion.update("happy", 0.05)
        self.assertEqual(res, FeedbackEvent.NO_ALERT)
        # Next frame with proximity should NOT trigger yet because buffer was cleared
        res = self.fusion.update("happy", 0.2)
        self.assertEqual(res, FeedbackEvent.NO_ALERT)

    def test_insufficient_majority(self):
        # 2 happy, 2 angry, 1 neutral -> no majority (>2.5 needed for 5)
        self.fusion.update("happy", 0.2)
        self.fusion.update("happy", 0.2)
        self.fusion.update("angry", 0.2)
        self.fusion.update("angry", 0.2)
        res = self.fusion.update("neutral", 0.2)
        self.assertEqual(res, FeedbackEvent.NO_ALERT)

if __name__ == "__main__":
    unittest.main()
