import os
from edge_assist.audio import AudioFeedback, _AUDIO_QUEUE

# FIX (Bug 2): HapticSimulator now exposes its AudioFeedback instance
# so callers can queue follow-up speech on the SAME worker thread,
# guaranteeing haptic-sim sound → speech ordering via the shared queue.

class HapticSimulator:
    def __init__(self):
        self.mode = os.getenv('HAPTIC_MODE', 'sim')
        if self.mode == 'sim':
            self.audio = AudioFeedback()
        else:
            self.audio = None

    def vibrate(self, pattern_name):
        """
        Executes a haptic pattern. On sim mode, queues the beep sound.
        Follow-up speech should be queued via self.audio.speak() by the caller
        immediately after vibrate() — the shared queue preserves ordering.
        """
        if self.mode == 'sim':
            if pattern_name == 'double-short':
                print('[HAPTIC] double-short')
                self.audio.speak("Beep. Beep.")
            elif pattern_name == 'long':
                print('[HAPTIC] long')
                self.audio.speak("Long beep.")
            elif pattern_name == 'single-short':
                print('[HAPTIC] single-short')
                self.audio.speak("Beep.")
            else:
                print(f'[HAPTIC] unknown pattern: {pattern_name}')
        else:
            # Pi GPIO logic (P6+)
            print(f'[HAPTIC-GPIO] Executing {pattern_name}')