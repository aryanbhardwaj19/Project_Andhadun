import time
from enum import Enum
from collections import deque, Counter
import threading
 
class FeedbackEvent(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    NO_ALERT = "no_alert"
 
# Emotions that map to NO_ALERT should NOT enter the vote buffer at all.
# Only real signal events are buffered so NO_ALERT can never win a majority vote.
_EMOTION_MAP = {
    "happy":   FeedbackEvent.POSITIVE,
    "angry":   FeedbackEvent.NEGATIVE,
    "disgust": FeedbackEvent.NEGATIVE,
    "fear":    FeedbackEvent.NEGATIVE,
    "sad":     FeedbackEvent.NEGATIVE,
    "neutral": FeedbackEvent.NEUTRAL,
    "surprise":FeedbackEvent.NEUTRAL,
}
 
class DecisionFusion:
    def __init__(self, buffer_size=5, cooldown_seconds=11.0, proximity_threshold=0.05):
        self.buffer = deque(maxlen=buffer_size)
        self.cooldown_seconds = cooldown_seconds
        self.proximity_threshold = proximity_threshold
        self.last_alert_time = 0
        self.lock = threading.Lock()
        self.cooldown_active = threading.Event()
 
    def update(self, emotion_label, face_area):
        with self.lock:
            # 1. Proximity gate
            if face_area < self.proximity_threshold:
                self.buffer.clear()
                return FeedbackEvent.NO_ALERT
 
            # 2. Map emotion → event.
            # FIX (Bug 1 & 3): unknown labels return NO_ALERT immediately
            # and are NEVER appended to the buffer.
            event = _EMOTION_MAP.get(emotion_label)
            if event is None:
                return FeedbackEvent.NO_ALERT
 
            self.buffer.append(event)
 
            # 3. Majority vote — only fires when buffer is full
            if len(self.buffer) < self.buffer.maxlen:
                return FeedbackEvent.NO_ALERT
 
            counts = Counter(self.buffer)
            # FIX (Bug 1): NO_ALERT is never in the buffer, so it can never win.
            # most_common(1)[0] now always returns a real signal event.
            most_common, count = counts.most_common(1)[0]
 
            if count < (self.buffer.maxlen // 2 + 1):
                return FeedbackEvent.NO_ALERT
 
            # 4. Cooldown gate
            current_time = time.time()
            if self.cooldown_active.is_set():
                if current_time - self.last_alert_time < self.cooldown_seconds:
                    return FeedbackEvent.NO_ALERT
                else:
                    self.cooldown_active.clear()
 
            self.last_alert_time = current_time
            self.cooldown_active.set()
            return most_common
 
    def reset_cooldown(self):
        with self.lock:
            self.last_alert_time = 0
            self.cooldown_active.clear()