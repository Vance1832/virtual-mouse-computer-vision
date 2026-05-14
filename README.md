# AI Virtual Mouse (Computer Vision)

This project is an experimental AI virtual mouse built using Python and computer vision.
It uses a webcam to track hand gestures and allows users to control mouse movement, clicks, scrolling, volume, and even launch applications without using a physical mouse or keyboard.

I built this project to practice computer vision, gesture recognition, and real-time interaction using OpenCV.

---

## Features

- Real-time hand tracking using OpenCV and cvzone
- Mouse movement controlled by index finger
- Left click, right click, and double click using hand gestures
- Scroll up / down using finger combinations
- Volume control gestures
- App launch mode (hold L-shape gesture, then select 1–5 fingers)
- Gesture stabilization and cooldowns to reduce false actions
- Modular and readable Python code structure

---

## Technologies Used

- Python
- OpenCV
- cvzone (HandTrackingModule)
- pyautogui
- NumPy

---

## Project Structure

virtual-mouse-computer-vision/
- gesture_controller.py   (Main application logic)
- find_camera.py          (Detects available camera)
- test_camera.py          (Camera testing script)
- config.json             (Gesture and camera configuration)
- apps_config.json        (App launch configuration)
- README.md

---

## How It Works (Simple Explanation)

1. The webcam captures live video.
2. Hand landmarks are detected using computer vision.
3. Different finger patterns are mapped to gestures.
4. Gestures are converted into mouse actions or system commands.
5. Cooldowns and smoothing are applied to improve stability.

---

## How to Run

1. Install required libraries using pip.
2. Run gesture_controller.py.
3. Press Q to quit the application.

---

## Notes

- This project is tested mainly on macOS.
- Lighting conditions and camera quality may affect accuracy.
- This is a learning project and still open for improvements.

---

## Motivation

I created this project as a computer science student to explore how computer vision can be used for human-computer interaction.
It helped me understand real-time systems, gesture recognition, and structuring larger Python projects.

---

## Author

Khant Zayar  
Computer Science Student – Rangsit University  
Email: khant.zayar.dev@gmail.com

