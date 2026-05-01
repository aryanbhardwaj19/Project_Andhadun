# 👁️ Project Andhadun: Edge-Assistive AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Phase_5_Complete-success.svg)]()
[![Hardware](https://img.shields.io/badge/Target-Raspberry_Pi_/_PC-orange.svg)]()

Project Andhadun is a high-performance, edge-based assistive system designed to aid the visually impaired. It utilizes computer vision to detect faces, classify emotions, and identify social roles in real-time, providing non-blocking audio and haptic feedback.

---

## 🏗️ Folder Structure

The project follows a modular architecture designed for high performance and easy porting to edge hardware (like Raspberry Pi).

```text
BUILD2/
├── edge_assist/                # Core logic package
│   ├── models/                 # Pre-trained TFLite and ONNX models
│   ├── tests/                  # Comprehensive test suite & demo data
│   │   ├── data/               # Labeled test images for validation
│   │   └── demo_output/        # Visual verification artifacts
│   ├── audio.py                # Asynchronous TTS & sound feedback
│   ├── emotion_classifier.py   # FER-based emotion detection (TFLite)
│   ├── face_detector.py        # YOLOv5n-based face localization
│   ├── fusion.py               # Decision logic, smoothing, & cooldowns
│   ├── haptic.py               # Haptic pattern simulation & GPIO control
│   ├── main_pipeline.py        # System integrator (The Brain)
│   └── role_detector.py        # Social role identification logic
├── .venv/                      # Isolated python environment
├── requirements.txt            # Project dependencies
└── yolov5nu.pt                 # YOLOv5 Nano Weights
```

---

## ✅ Project Completion Status

The project is currently in its **Final Integration Phase**. All core modules are implemented and verified.

| Module | Status | Description |
| :--- | :---: | :--- |
| **Face Detection** | 🟢 | Real-time localization using YOLOv5n. Optimized for low latency. |
| **Emotion Classification** | 🟢 | 7-class emotion detection (FER) with TFLite backend. |
| **Role Detection** | 🟢 | Proximity-based and visual role identification. |
| **Decision Fusion** | 🟢 | Window-based voting (5 frames) and alert cooldown (3s). |
| **Audio Feedback** | 🟢 | Non-blocking pyttsx3 integration for natural voice alerts. |
| **Haptic Feedback** | 🟢 | Pattern-based vibration feedback (Simulated/GPIO). |
| **Main Pipeline** | 🟢 | Full end-to-end integration and system loop. |

---

## 🚀 Core Features

- **Edge-First Design**: Optimized for low-power devices with TFLite and Nano-model weights.
- **Robust Decision Fusion**: Prevents alert flooding through temporal smoothing and proximity gating.
- **Non-Blocking Feedback**: System remains responsive during audio/haptic playback.
- **Automated Validation**: 90%+ test coverage across core modules to ensure reliability.

## 🛠️ Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   ```bash
   python -m edge_assist.main_pipeline
   ```

3. **Run Tests**:
   ```bash
   python -m unittest discover edge_assist/tests
   ```

---

## 👥 Contributors

- **Anirudha Mohanty** - *Project Lead & Architect*
- **Aryan Bharadwaj** - *Lead Developer & Visionary*

*Designed with ❤️ for Accessibility.*
