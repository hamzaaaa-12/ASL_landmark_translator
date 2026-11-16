
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import time
import threading

from collections import deque, Counter
from pathlib import Path

from streamlit_webrtc import VideoProcessorBase

import handlandmark


# ============================================================
# 1. PATHS AND MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "asl_knn_model.pkl"
HAND_MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]

if len(feature_columns) != 30:
    raise ValueError(
        f"Expected 30 features, but model expects {len(feature_columns)}"
    )

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Hand Landmarker model not found: {HAND_MODEL_PATH}"
    )


# ============================================================
# 2. SETTINGS
# ============================================================

LETTER_HOLD_TIME = 1.5

PINCH_HOLD_TIME = 0.5
PINCH_THRESHOLD = 0.35

SWIPE_THRESHOLD = 0.20
SWIPE_MAX_DURATION = 0.35
SWIPE_COOLDOWN = 1.0

PREDICTION_HISTORY_SIZE = 7
MIN_STABLE_VOTES = 5

STABILITY_GRACE_TIME = 0.25


# ============================================================
# 3. HAND LANDMARK DRAWING
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand_landmarks(frame, landmarks):

    height, width, _ = frame.shape
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1
        )


# ============================================================
# 4. ASL VIDEO PROCESSOR
# ============================================================

class ASLVideoProcessor(VideoProcessorBase):

    def __init__(self):

        # -------------------------------
        # Text
        # -------------------------------

        self.sentence = ""

        # -------------------------------
        # Thread safety
        # -------------------------------

        self.lock = threading.Lock()

        # -------------------------------
        # Prediction
        # -------------------------------

        self.prediction_history = deque(
            maxlen=PREDICTION_HISTORY_SIZE
        )

        self.current_prediction = "None"
        self.status_message = "Show your hand"

        # -------------------------------
        # Letter stabilization
        # -------------------------------

        self.candidate_letter = None
        self.candidate_start_time = None
        self.last_stable_time = None
        self.letter_committed = False

        # -------------------------------
        # Pinch
        # -------------------------------

        self.pinch_start_time = None
        self.pinch_committed = False

        # -------------------------------
        # Swipe
        # -------------------------------

        self.wrist_history = deque(maxlen=30)
        self.last_swipe_time = 0

        # -------------------------------
        # MediaPipe timestamp
        # -------------------------------

        self.last_timestamp_ms = 0

        # -------------------------------
        # MediaPipe setup
        # -------------------------------

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = HandLandmarker.create_from_options(
            options
        )

    # ========================================================
    # 5. TEXT CONTROLS
    # ========================================================

    # def add_letter(self, letter):

    #     self.sentence += str(letter)

    # def add_space(self):

    #     if self.sentence and not self.sentence.endswith(" "):
    #         self.sentence += " "

    # def delete_last(self):

    #     self.sentence = self.sentence[:-1]

    # def clear_text(self):

    #     with self.lock:
    #         self.sentence = ""

    def add_letter(self, letter):
        with self.lock:
            self.sentence += str(letter)


    def add_space(self):
        with self.lock:
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "


    def delete_last(self):
        with self.lock:
            self.sentence = self.sentence[:-1]


    def clear_text(self):
        with self.lock:
            self.sentence = ""

    def get_state(self):

        with self.lock:
            return {
                "sentence": self.sentence,
                "prediction": self.current_prediction,
                "status": self.status_message
            }
    # ========================================================
    # 6. PINCH DETECTION
    # ========================================================

    def detect_pinch(self, landmarks):

        thumb = landmarks[4]
        index = landmarks[8]

        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        pinch_distance = (
            (thumb.x - index.x) ** 2 +
            (thumb.y - index.y) ** 2
        ) ** 0.5

        palm_size = (
            (wrist.x - middle_mcp.x) ** 2 +
            (wrist.y - middle_mcp.y) ** 2
        ) ** 0.5

        if palm_size < 0.0001:
            return False, 0

        normalized_distance = pinch_distance / palm_size

        is_pinching = normalized_distance < PINCH_THRESHOLD

        return is_pinching, normalized_distance

    def process_pinch(self, is_pinching):

        current_time = time.monotonic()

        if not is_pinching:

            self.pinch_start_time = None
            self.pinch_committed = False

            return "Pinch Released"

        if self.pinch_start_time is None:

            self.pinch_start_time = current_time
            self.pinch_committed = False

        elapsed = current_time - self.pinch_start_time

        if (
            elapsed >= PINCH_HOLD_TIME
            and not self.pinch_committed
        ):

            self.add_space()
            self.pinch_committed = True

            return "SPACE ADDED"

        if self.pinch_committed:
            return "SPACE ADDED"

        return f"Pinch: {elapsed:.1f}s"

    # ========================================================
    # 7. LEFT SWIPE DETECTION
    # ========================================================

    def detect_left_swipe(self, landmarks):

        current_time = time.monotonic()

        wrist_x = landmarks[0].x

        self.wrist_history.append(
            (current_time, wrist_x)
        )

        if len(self.wrist_history) < 3:
            return False

        valid_points = [
            point for point in self.wrist_history
            if current_time - point[0] <= SWIPE_MAX_DURATION
        ]

        if len(valid_points) < 3:
            return False

        start_time, start_x = valid_points[0]
        end_time, end_x = valid_points[-1]

        duration = end_time - start_time

        # Webcam image is flipped.
        # Visual LEFT movement means x decreases.

        movement = start_x - end_x

        if (
            movement >= SWIPE_THRESHOLD
            and duration <= SWIPE_MAX_DURATION
            and current_time - self.last_swipe_time >= SWIPE_COOLDOWN
        ):

            self.last_swipe_time = current_time
            self.wrist_history.clear()

            return True

        return False

    # ========================================================
    # 8. PREDICTION STABILIZATION
    # ========================================================

    def get_stable_prediction(self, prediction):

        self.prediction_history.append(prediction)

        valid_predictions = [
            p for p in self.prediction_history
            if p is not None
        ]

        if not valid_predictions:
            return None

        counts = Counter(valid_predictions)

        most_common_letter, votes = counts.most_common(1)[0]

        if votes >= MIN_STABLE_VOTES:
            return most_common_letter

        return None

    def reset_letter_selection(self):

        self.candidate_letter = None
        self.candidate_start_time = None
        self.last_stable_time = None
        self.letter_committed = False

    def process_letter_selection(self, stable_prediction):

        current_time = time.monotonic()

        if stable_prediction is None:

            if (
                self.last_stable_time is not None
                and current_time - self.last_stable_time
                > STABILITY_GRACE_TIME
            ):
                self.reset_letter_selection()

            return "Waiting for stable letter"

        if stable_prediction != self.candidate_letter:

            self.candidate_letter = stable_prediction
            self.candidate_start_time = current_time
            self.last_stable_time = current_time
            self.letter_committed = False

        else:

            self.last_stable_time = current_time

        elapsed = current_time - self.candidate_start_time

        if (
            elapsed >= LETTER_HOLD_TIME
            and not self.letter_committed
        ):

            self.add_letter(self.candidate_letter)
            self.letter_committed = True

            return f"Added: {self.candidate_letter}"

        if self.letter_committed:
            return f"Committed: {self.candidate_letter}"

        return f"Hold {self.candidate_letter}: {elapsed:.1f}s"

    # ========================================================
    # 9. PROCESS EACH WEBCAM FRAME
    # ========================================================

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        # Flip webcam image
        image = cv2.flip(image, 1)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # MediaPipe VIDEO mode requires increasing timestamps
        timestamp_ms = int(time.monotonic() * 1000)

        timestamp_ms = max(
            timestamp_ms,
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        results = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        self.current_prediction = "None"

        # ====================================================
        # HAND DETECTED
        # ====================================================

        if results.hand_landmarks:

            landmarks = results.hand_landmarks[0]

            draw_hand_landmarks(image, landmarks)

            is_pinching, pinch_distance = self.detect_pinch(
                landmarks
            )

            # --------------------------------------------
            # A. PINCH HAS PRIORITY
            # --------------------------------------------

            if is_pinching:

                self.wrist_history.clear()
                self.prediction_history.clear()
                self.reset_letter_selection()

                self.status_message = self.process_pinch(True)

            else:

                self.process_pinch(False)

                # ----------------------------------------
                # B. LEFT SWIPE
                # ----------------------------------------

                swipe_detected = self.detect_left_swipe(
                    landmarks
                )

                if swipe_detected:

                    self.delete_last()

                    self.prediction_history.clear()
                    self.reset_letter_selection()

                    self.current_prediction = "None"
                    self.status_message = "DELETE"

                else:

                    # ------------------------------------
                    # C. EXTRACT 30 FEATURES
                    # ------------------------------------

                    features = handlandmark.extract_features(
                        landmarks
                    )

                    if features is not None:

                        if len(features) != 30:
                            raise ValueError(
                                f"Expected 30 features, got {len(features)}"
                            )

                        feature_df = pd.DataFrame(
                            [features],
                            columns=feature_columns
                        )

                        prediction = model.predict(
                            feature_df
                        )[0]

                        self.current_prediction = prediction

                        stable_prediction = (
                            self.get_stable_prediction(prediction)
                        )

                        self.status_message = (
                            self.process_letter_selection(
                                stable_prediction
                            )
                        )

                    else:

                        self.prediction_history.clear()
                        self.reset_letter_selection()

                        self.status_message = (
                            "Feature extraction failed"
                        )

        # ====================================================
        # NO HAND DETECTED
        # ====================================================

        else:

            self.prediction_history.clear()
            self.wrist_history.clear()

            self.process_pinch(False)
            self.reset_letter_selection()

            self.current_prediction = "None"
            self.status_message = "Show your hand"

        # ====================================================
        # DRAW INFORMATION ON VIDEO
        # ====================================================

        with self.lock:

            sentence_copy = self.sentence
            prediction_copy = self.current_prediction
            status_copy = self.status_message

        cv2.putText(
            image,
            f"Prediction: {prediction_copy}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Status: {status_copy}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Text: {sentence_copy}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            "Pinch: SPACE | Left Swipe: DELETE",
            (20, image.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        return type(frame).from_ndarray(
            image,
            format="bgr24"
        )

    def __del__(self):

        try:
            self.landmarker.close()
        except Exception:
            pass
