import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import numpy as np
import json
import logging


from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
from pathlib import Path
import subprocess
import platform

pyautogui.FAILSAFE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('virtual_mouse.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================
@dataclass
class GestureConfig:
    detection_confidence: float = 0.8
    max_hands: int = 1
    smoothing_factor: int = 4
    camera_width: int = 640
    camera_height: int = 480
    margin: int = 90

    # Cooldowns
    click_cooldown: float = 1.0
    scroll_cooldown: float = 0.7
    volume_cooldown: float = 0.4
    swipe_cooldown: float = 1.0
    app_launch_cooldown: float = 2.0

    # Scroll amount
    scroll_amount: int = 7

    # App Mode
    app_mode_duration: float = 2.0
    app_mode_expire: float = 5.0


# =========================
# APP LAUNCHER
# =========================
class AppLauncher:
    def __init__(self, config_path="apps_config.json"):
        self.config_path = Path(config_path)
        self.system = platform.system()
        self.apps = self._load_apps()

    def _load_apps(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except:
                pass

        default_apps = {
            "spotify": {"Darwin": "open -a Spotify"},
            "chrome": {"Darwin": "open -a 'Google Chrome'"},
            "vscode": {"Darwin": "open -a 'Visual Studio Code'"},
            "calculator": {"Darwin": "open -a Calculator"},
            "notepad": {"Darwin": "open -a TextEdit"}
        }
        return default_apps

    def launch(self, app_name):
        app_name = app_name.lower()
        if app_name not in self.apps:
            return False
        try:
            cmd = self.apps[app_name][self.system]
            subprocess.Popen(cmd, shell=True)
            return True
        except:
            return False


# =========================
# CAMERA MANAGER
# =========================
class CameraManager:
    @staticmethod
    def find_camera():
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    return cap
                cap.release()
        return None

    @staticmethod
    def configure(cap, w, h):
        cap.set(3, w)
        cap.set(4, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return True
    
# =========================
# GESTURE RECOGNIZER (IMPROVED)
# =========================
class GestureRecognizer:
    def __init__(self, cfg: GestureConfig):
        self.cfg = cfg
        self.gesture_history = []
        self.history_size = 6
        self.min_conf = 3

        # App mode
        self.app_mode_active = False
        self.app_mode_start = 0
        self.app_mode_progress = 0
        self.app_expire_time = 0
        self.app_hold_start = 0
        self.last_app_fingers = None

    # ------------------------------------
    # Stabilize gestures using history
    # ------------------------------------
    def _push_history(self, g):
        self.gesture_history.append(g)
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)

    def _stable(self):
        if len(self.gesture_history) < self.min_conf:
            return self.gesture_history[-1] if self.gesture_history else "NONE"

        counts = {}
        for g in self.gesture_history:
            counts[g] = counts.get(g, 0) + 1

        best = max(counts.items(), key=lambda x: x[1])
        if best[1] >= self.min_conf:
            return best[0]
        return self.gesture_history[-1]

    # ------------------------------------
    # Helpers
    # ------------------------------------
    def _is_fist(self, fingers):
        return fingers == [0, 0, 0, 0, 0]

    def _is_pinch(self, lm, fingers, frame_h):
        if fingers[0] != 1 or fingers[1] != 1:
            return False

        thumb = np.array(lm[4])
        index = np.array(lm[8])
        dist = np.linalg.norm(thumb - index)

        threshold = max(18, frame_h * 0.055)
        return dist < threshold

    def _is_l_shape(self, fingers, lm):
        if len(lm) < 9:
            return False
        if not (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0):
            return False

        thumb = np.array(lm[4])
        index = np.array(lm[8])
        wrist = np.array(lm[0])

        v1 = thumb - wrist
        v2 = index - wrist

        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))

        return 50 < ang < 120

    # Improved swipe
    def _swipe_direction(self, lm, fingers):
        if fingers != [1, 0, 0, 0, 0]:  # thumb only
            return None
        wrist = lm[0]
        thumb = lm[4]

        dy = abs(thumb[1] - wrist[1])
        if dy > 80:  # too vertical
            return None

        if thumb[0] < wrist[0] - 20:
            return "LEFT"
        if thumb[0] > wrist[0] + 20:
            return "RIGHT"
        return None

    # ------------------------------------
    # MAIN GESTURE DETECTOR
    # ------------------------------------
    def detect(self, fingers, hand):
        lm = hand["lmList"]
        frame_h = hand.get("frameHeight", 480)
        now = time.time()

        gesture = "NONE"
        finger_count = sum(fingers)
        app_number = None

        # --------------------------
        # APP MODE: L-shape hold
        # --------------------------
        if self._is_l_shape(fingers, lm):
            if self.app_mode_start == 0:
                self.app_mode_start = now

            held = now - self.app_mode_start
            self.app_mode_progress = min(1.0, held / self.cfg.app_mode_duration)

            if held >= self.cfg.app_mode_duration and not self.app_mode_active:
                self.app_mode_active = True
                self.app_expire_time = now + self.cfg.app_mode_expire
                gesture = "APP_MODE_ACTIVATED"
        else:
            self.app_mode_start = 0
            self.app_mode_progress = 0

        # --------------------------
        # If app mode active → choose 1-5
        # --------------------------
        if self.app_mode_active:
            if now > self.app_expire_time:
                self.app_mode_active = False

            # Prevent accidental app selection when still in L-shape
            if fingers == [1, 1, 0, 0, 0]:
                gesture = "APP_MODE_ACTIVE"
                app_number = None
                self.last_app_fingers = None
                return gesture, finger_count, app_number

            if 1 <= finger_count <= 5:
                if self.last_app_fingers != finger_count:
                    self.app_hold_start = now
                    self.last_app_fingers = finger_count

                if now - self.app_hold_start >= 1.5:
                    app_number = finger_count
                    gesture = f"LAUNCH_APP_{app_number}"
                    self.app_mode_active = False
                else:
                    gesture = f"HOLD_{finger_count}_FOR_APP"
            else:
                gesture = "APP_MODE_ACTIVE"

        # --------------------------
        # Right Click = Pinch
        # --------------------------
        if not self.app_mode_active:
            if self._is_pinch(lm, fingers, frame_h):
                gesture = "RIGHT_CLICK"

        # --------------------------
        # SWIPE
        # --------------------------
        if not self.app_mode_active:
            sw = self._swipe_direction(lm, fingers)
            if sw == "LEFT":
                gesture = "SWIPE_LEFT"
            elif sw == "RIGHT":
                gesture = "SWIPE_RIGHT"

        # --------------------------
        # NORMAL MOUSE GESTURES
        # --------------------------
        if gesture == "NONE" and not self.app_mode_active:
            # Scroll
            if finger_count == 2 and fingers[1] == 1 and fingers[2] == 1:
                gesture = "SCROLL_UP"

            elif finger_count == 3:
                gesture = "SCROLL_DOWN"

            # Fist = LEFT CLICK (improved)
            elif self._is_fist(fingers):
                gesture = "LEFT_CLICK"

            #  Rock sign = double click
            elif fingers == [1, 1, 0, 0, 1]:
                gesture = "DOUBLE_CLICK"

            # Volume gestures
            elif fingers == [0, 0, 0, 1, 0]:
                gesture = "VOLUME_DOWN"
            elif fingers == [0, 0, 0, 0, 1]:
                gesture = "VOLUME_UP"

            # MOVE = EXACTLY one finger (index) up
            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "MOVE"

        # ---- IMPORTANT: Do NOT stabilize app launch gestures ----
        if isinstance(gesture, str) and gesture.startswith("LAUNCH_APP_"):
            return gesture, finger_count, app_number

        # push into history
        self._push_history(gesture)

        # use stable gesture
        stable = self._stable()

        return stable, finger_count, app_number
    
# =========================
# ACTION EXECUTOR (IMPROVED)
# =========================
class ActionExecutor:
    def __init__(self, cfg: GestureConfig, app_launcher: AppLauncher):
        self.cfg = cfg
        self.app_launcher = app_launcher

        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x = 0
        self.prev_y = 0
        # Ultra-smooth cursor variables
        self.smooth_x = 0
        self.smooth_y = 0
        self.deadzone = 5      # ignore small shakes
        self.alpha = 0.25      # smoothing strength

        # Cooldowns
        self.last_click = 0
        self.last_scroll = 0
        self.last_volume = 0
        self.last_swipe = 0
        self.last_app_launch = 0

        # App shortcuts
        self.app_shortcuts = {
            1: "spotify",
            2: "chrome",
            3: "vscode",
            4: "calculator",
            5: "notepad"
        }

    # ------------------------------------
    # Separate cooldowns
    # ------------------------------------
    def _cool(self, last, cd):
        return (time.time() - last) >= cd

    # ------------------------------------
    # Smooth cursor mapping
    # ------------------------------------
    def _map_coords(self, x, y, fw, fh):
        # Convert camera coords → screen coords. np.interp for linear mapping
        target_x = np.interp(x, [self.cfg.margin, fw - self.cfg.margin], [0, self.screen_w])
        target_y = np.interp(y, [self.cfg.margin, fh - self.cfg.margin], [0, self.screen_h])

        # Deadzone to reduce jitter
        if abs(target_x - self.smooth_x) < self.deadzone:
            target_x = self.smooth_x
        if abs(target_y - self.smooth_y) < self.deadzone:
            target_y = self.smooth_y

        # Exponential smoothing
        self.smooth_x = (self.alpha * target_x) + ((1 - self.alpha) * self.smooth_x)
        self.smooth_y = (self.alpha * target_y) + ((1 - self.alpha) * self.smooth_y)

        return int(self.smooth_x), int(self.smooth_y)

    # ------------------------------------
    # EXECUTE GESTURES
    # ------------------------------------
    def execute(self, gesture, hand, fw, fh, app_number):
        # Reset previous cursor position when not moving to prevent jump
        if gesture != "MOVE":
            self.prev_x = self.prev_x
            self.prev_y = self.prev_y
        # Check for QUIT_APP gesture
        if gesture == "QUIT_APP":
            print("Middle finger detected: Quitting app...")
            import sys
            sys.exit(0)

        now = time.time()

        # ------------------------------------
        # App Launch
        # ------------------------------------
        if gesture.startswith("LAUNCH_APP_") and app_number:
            if self._cool(self.last_app_launch, self.cfg.app_launch_cooldown):
                self.last_app_launch = now

                app_name = self.app_shortcuts.get(app_number)
                if app_name:
                    ok = self.app_launcher.launch(app_name)
                    return f"Launching {app_name}..." if ok else f"Failed to launch {app_name}"
            return "App Launch Cooldown..."

        if gesture == "APP_MODE_ACTIVATED":
            return "APP MODE ACTIVE — hold 1–5 fingers"

        if gesture.startswith("HOLD_"):
            return f"Holding: {gesture}"

        if gesture == "APP_MODE_ACTIVE":
            return "APP MODE — choose 1–5"

        # ------------------------------------
        # MOVE
        # ------------------------------------
        if gesture == "MOVE" and hand:
            lm = hand["lmList"]
            if len(lm) > 8:
                ix, iy = lm[8][0], lm[8][1]
                # Only update mapping when index finger is steady
                mx, my = self._map_coords(ix, iy, fw, fh)
                self.prev_x, self.prev_y = mx, my
                pyautogui.moveTo(mx, my, _pause=False)
                return f"Move ({mx},{my})"

        # ------------------------------------
        # LEFT CLICK
        # ------------------------------------
        if gesture == "LEFT_CLICK":
            if self._cool(self.last_click, self.cfg.click_cooldown):
                self.last_click = now
                # Freeze cursor during click
                pyautogui.moveTo(self.prev_x, self.prev_y, _pause=False)
                pyautogui.click()
                return "Left Click"
            return "Click cooldown"

        # ------------------------------------
        # RIGHT CLICK
        # ------------------------------------
        if gesture == "RIGHT_CLICK":
            if self._cool(self.last_click, self.cfg.click_cooldown):
                self.last_click = now
                pyautogui.rightClick()
                return "Right Click"
            return "Click cooldown"

        # ------------------------------------
        # DOUBLE CLICK (🤘)
        # ------------------------------------
        if gesture == "DOUBLE_CLICK":
            if self._cool(self.last_click, self.cfg.click_cooldown):
                self.last_click = now
                pyautogui.doubleClick()
                return "Double Click"
            return "Double Click Cooldown"

        # ------------------------------------
        # SCROLL
        # ------------------------------------
        if gesture == "SCROLL_UP":
            if self._cool(self.last_scroll, self.cfg.scroll_cooldown):
                self.last_scroll = now
                pyautogui.scroll(+self.cfg.scroll_amount)
                return "Scroll Up"
            return "Scroll cooldown"

        if gesture == "SCROLL_DOWN":
            if self._cool(self.last_scroll, self.cfg.scroll_cooldown):
                self.last_scroll = now
                pyautogui.scroll(-self.cfg.scroll_amount)
                return "Scroll Down"
            return "Scroll cooldown"

        # ------------------------------------
        # VOLUME
        # ------------------------------------
        if gesture == "VOLUME_UP":
            if self._cool(self.last_volume, self.cfg.volume_cooldown):
                self.last_volume = now
                subprocess.run(["osascript", "-e",
                                "set volume output volume ((output volume of (get volume settings)) + 5)"])
                return "Volume Up"

        if gesture == "VOLUME_DOWN":
            if self._cool(self.last_volume, self.cfg.volume_cooldown):
                self.last_volume = now
                subprocess.run(["osascript", "-e",
                                "set volume output volume ((output volume of (get volume settings)) - 5)"])
                return "Volume Down"

        # ------------------------------------
        # SWIPE (Desktop switch)
        # ------------------------------------
        if gesture == "SWIPE_LEFT":
            if self._cool(self.last_swipe, self.cfg.swipe_cooldown):
                self.last_swipe = now
                pyautogui.hotkey("ctrl", "left")
                return "Swipe Left"
            return "Swipe cooldown"

        if gesture == "SWIPE_RIGHT":
            if self._cool(self.last_swipe, self.cfg.swipe_cooldown):
                self.last_swipe = now
                pyautogui.hotkey("ctrl", "right")
                return "Swipe Right"
            return "Swipe cooldown"

        return ""
        

# =========================
# GESTURE CONTROLLER
# =========================
class GestureController:
    def __init__(self):
        self.cfg = GestureConfig()
        self.detector = HandDetector(maxHands=self.cfg.max_hands,
                                     detectionCon=self.cfg.detection_confidence)
        self.app_launcher = AppLauncher()
        self.executor = ActionExecutor(self.cfg, self.app_launcher)
        self.recognizer = GestureRecognizer(self.cfg)

    # ------------------------------------
    # UI TEXT
    # ------------------------------------
    def _draw_text(self, frame, gesture, action, fingers):
        cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Action: {action}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(frame, f"Fingers: {fingers}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        # App progress bar
        if 0 < self.recognizer.app_mode_progress < 1:
            h, w = frame.shape[:2]
            bar_w = int(w * 0.4)
            filled = int(bar_w * self.recognizer.app_mode_progress)
            y = 150

            cv2.rectangle(frame, (10, y), (10 + bar_w, y + 12), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, y), (10 + filled, y + 12), (0, 255, 0), -1)
            cv2.putText(frame, "App Mode...", (10, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ------------------------------------
    # MAIN LOOP
    # ------------------------------------
    def run(self):
        cap = CameraManager.find_camera()
        if not cap:
            print("❌ No camera detected.")
            return

        CameraManager.configure(cap, self.cfg.camera_width, self.cfg.camera_height)

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            hands, frame = self.detector.findHands(frame, flipType=True)

            gesture = "NONE"
            action = ""
            fingers = []

            if hands:
                hand = hands[0]
                hand["frameHeight"] = self.cfg.camera_height

                fingers = self.detector.fingersUp(hand)
                gesture, _, app_num = self.recognizer.detect(fingers, hand)
                action = self.executor.execute(gesture, hand,
                                               self.cfg.camera_width,
                                               self.cfg.camera_height,
                                               app_num)

            self._draw_text(frame, gesture, action, fingers)

            cv2.imshow("AI Virtual Mouse", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================
# MAIN
# =========================
def main():
    print("AI Virtual Mouse — Improved Version")
    controller = GestureController()
    controller.run()


if __name__ == "__main__":
    main()