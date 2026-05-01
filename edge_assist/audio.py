import pyttsx3
import threading
import queue
import time

_AUDIO_QUEUE = queue.Queue()
TTS_LOCK = threading.Lock()

def _worker():
    while True:
        text = _AUDIO_QUEUE.get()
        if text is None: break
        try:
            # Fresh engine per item for maximum reliability on Windows threads
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
            
            print(f"[AUDIO WORKER] Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
            
            # Explicit cleanup
            del engine
            
            _AUDIO_QUEUE.task_done()
            print(f"[AUDIO WORKER] Done: {text}")
            time.sleep(0.1) # Stability gap
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
            _AUDIO_QUEUE.task_done()

_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()

class AudioFeedback:
    def speak(self, text):
        _AUDIO_QUEUE.put(text)

if __name__ == "__main__":
    audio = AudioFeedback()
    print("Testing unified audio worker...")
    audio.speak("Testing system")
    audio.speak("One two three")
    time.sleep(3)