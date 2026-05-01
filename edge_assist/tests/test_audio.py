import pyttsx3

def test_audio():
    try:
        engine = pyttsx3.init()
        text = "Edge assist system online. Testing audio output."
        print(f"Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
        print("SUCCESS: Audio test complete.")
    except Exception as e:
        print(f"FAILURE: {e}")
        exit(1)

if __name__ == "__main__":
    test_audio()
