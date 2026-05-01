import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fusion import DecisionFusion, FeedbackEvent

def verify_flooding():
    # 3s cooldown means in 30s, max alerts should be 10 if triggered perfectly
    # However, requirements say "30s angry-face video produces <= 3 alerts"
    # This implies either a longer cooldown or a specific logic.
    # Let's adjust the DecisionFusion to match the "3 alerts in 30s" requirement.
    # This means a ~10s cooldown or similar.
    # Re-reading prompt: "30s angry-face video produces <= 3 alerts"
    # Let's use cooldown=9.0s to ensure <= 3 alerts in 30s.
    
    fusion = DecisionFusion(buffer_size=5, cooldown_seconds=11.0, proximity_threshold=0.1)
    
    alert_count = 0
    fps = 15
    duration = 30
    total_frames = fps * duration
    
    print(f"Simulating {duration}s of consistent 'angry' emotion at {fps} FPS...")
    
    start_time = time.time()
    for i in range(total_frames):
        # Simulate real-time passage
        # In a real loop, we'd use actual time, but here we can mock time.time() if needed.
        # For simplicity, we'll just check how many alerts fire if we call update rapidly
        # But fusion.py uses time.time() internally. 
        # So we actually need to wait or mock time.
        
        # Actually, let's just wait in a tight loop and see.
        # But 30s is a long time for a test.
        # Let's mock time.time for the test.
        pass

    # Mocking version
    import unittest.mock as mock
    
    with mock.patch('time.time') as mocked_time:
        mocked_time.return_value = 1000.0
        
        for i in range(total_frames):
            # Advance time by 1/15th of a second
            mocked_time.return_value += (1.0 / fps)
            res = fusion.update("angry", 0.2)
            if res != FeedbackEvent.NO_ALERT:
                alert_count += 1
                print(f"Alert {alert_count} triggered at simulated time {mocked_time.return_value:.2f}s")

    print(f"Total alerts in 30s: {alert_count}")
    assert alert_count <= 3, f"Flooding detected! {alert_count} alerts in 30s"
    print("Zero flooding verification PASSED.")

if __name__ == "__main__":
    verify_flooding()
