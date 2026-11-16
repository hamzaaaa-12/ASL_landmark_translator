import cv2
import mediapipe as mp
import joblib
import pandas as pd
import time

from collections import deque, Counter
from pathlib import Path

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

print("Model loaded successfully.")
print("Number of features:", len(feature_columns))


# ============================================================
# 2. SETTINGS
# ============================================================

# Letter recognition
LETTER_HOLD_TIME = 0.75

# Space gesture: Thumb + Index Pinch
PINCH_HOLD_TIME = 0.5
PINCH_THRESHOLD = 0.35

# Delete gesture: Left Swipe
SWIPE_THRESHOLD = 0.20
SWIPE_MAX_DURATION = 0.65
SWIPE_COOLDOWN = 1.0

# Prediction stabilization
PREDICTION_HISTORY_SIZE = 7
MIN_STABLE_VOTES = 5

# Tolerate short prediction interruptions
STABILITY_GRACE_TIME = 0.25


# ============================================================
# 3. TEXT BUILDER
# ============================================================

sentence = ""


def add_letter(letter):
    global sentence

    sentence += letter
    print("Text:", sentence)


def add_space():
    global sentence

    if sentence and not sentence.endswith(" "):
        sentence += " "

    print("Text:", sentence)


def delete_last():
    global sentence

    sentence = sentence[:-1]
    print("Text:", sentence)


def clear_text():
    global sentence

    sentence = ""
    print("Text cleared.")


# ============================================================
# 4. DRAW HAND LANDMARKS
# ============================================================

# Connections between the 21 MediaPipe landmarks
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

    # Draw connections
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Draw landmarks
    for point in points:

        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1
        )


# ============================================================
# 5. PINCH DETECTION
# ============================================================

def detect_pinch(landmarks):

    # Thumb Tip = Landmark 4
    # Index Finger Tip = Landmark 8

    thumb = landmarks[4]
    index = landmarks[8]

    # Wrist = Landmark 0
    # Middle Finger MCP = Landmark 9

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    # Distance between thumb and index
    pinch_distance = (
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2
    ) ** 0.5

    # Normalize using palm size
    palm_size = (
        (wrist.x - middle_mcp.x) ** 2 +
        (wrist.y - middle_mcp.y) ** 2
    ) ** 0.5

    if palm_size < 0.0001:
        return False, 0

    normalized_distance = pinch_distance / palm_size

    is_pinching = normalized_distance < PINCH_THRESHOLD

    return is_pinching, normalized_distance


# ============================================================
# 6. PINCH HOLD CONTROLLER
# ============================================================

pinch_start_time = None
pinch_committed = False


def process_pinch(is_pinching):

    global pinch_start_time
    global pinch_committed

    current_time = time.monotonic()

    if not is_pinching:

        pinch_start_time = None
        pinch_committed = False

        return False, "Pinch Released"

    if pinch_start_time is None:

        pinch_start_time = current_time
        pinch_committed = False

    elapsed = current_time - pinch_start_time

    if elapsed >= PINCH_HOLD_TIME and not pinch_committed:

        add_space()

        pinch_committed = True

        return True, "SPACE ADDED"

    if pinch_committed:

        return True, "SPACE ADDED"

    return True, f"Pinch: {elapsed:.1f}s"


# ============================================================
# 7. LEFT SWIPE DETECTION
# ============================================================

wrist_history = deque(maxlen=30)

last_swipe_time = 0


def detect_left_swipe(landmarks):

    global last_swipe_time

    current_time = time.monotonic()

    # Wrist landmark
    wrist_x = landmarks[0].x

    wrist_history.append((current_time, wrist_x))

    if len(wrist_history) < 3:
        return False

    # Find the earliest point inside the allowed time window
    valid_points = [
        point for point in wrist_history
        if current_time - point[0] <= SWIPE_MAX_DURATION
    ]

    if len(valid_points) < 3:
        return False

    start_time, start_x = valid_points[0]
    end_time, end_x = valid_points[-1]

    duration = end_time - start_time

    # On the flipped webcam image:
    # Moving visually LEFT means x decreases.

    movement = start_x - end_x

    if (
        movement >= SWIPE_THRESHOLD
        and duration <= SWIPE_MAX_DURATION
        and current_time - last_swipe_time >= SWIPE_COOLDOWN
    ):

        last_swipe_time = current_time

        wrist_history.clear()

        return True

    return False


# ============================================================
# 8. LETTER STABILIZATION
# ============================================================

prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

candidate_letter = None
candidate_start_time = None
last_stable_time = None
letter_committed = False


def get_stable_prediction(prediction):

    prediction_history.append(prediction)

    valid_predictions = [
        p for p in prediction_history
        if p is not None
    ]

    if not valid_predictions:
        return None

    counts = Counter(valid_predictions)

    most_common_letter, votes = counts.most_common(1)[0]

    if votes >= MIN_STABLE_VOTES:
        return most_common_letter

    return None


def reset_letter_selection():

    global candidate_letter
    global candidate_start_time
    global last_stable_time
    global letter_committed

    candidate_letter = None
    candidate_start_time = None
    last_stable_time = None
    letter_committed = False


def process_letter_selection(stable_prediction):

    global candidate_letter
    global candidate_start_time
    global last_stable_time
    global letter_committed

    current_time = time.monotonic()

    # No stable prediction
    if stable_prediction is None:

        if (
            last_stable_time is not None
            and current_time - last_stable_time > STABILITY_GRACE_TIME
        ):

            reset_letter_selection()

        return "Waiting for stable letter"

    # A new letter appeared
    if stable_prediction != candidate_letter:

        candidate_letter = stable_prediction

        candidate_start_time = current_time

        last_stable_time = current_time

        letter_committed = False

    else:

        last_stable_time = current_time

    elapsed = current_time - candidate_start_time

    # Add letter once
    if elapsed >= LETTER_HOLD_TIME and not letter_committed:

        add_letter(candidate_letter)

        letter_committed = True

        return f"Added: {candidate_letter}"

    if letter_committed:

        return f"Committed: {candidate_letter}"

    return f"Hold {candidate_letter}: {elapsed:.1f}s"


# ============================================================
# 9. MEDIAPIPE TASKS SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

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


# ============================================================
# 10. MAIN LOOP
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")


last_timestamp_ms = 0

current_prediction = "None"
status_message = "Show your hand"


try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Cannot read webcam frame.")
                break

            # Flip webcam image
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # MediaPipe image
            mp_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=rgb_frame
            )

            # Timestamp must increase
            timestamp_ms = int(time.monotonic() * 1000)

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # Detect hand
            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            current_prediction = "None"

            # =================================================
            # HAND DETECTED
            # =================================================

            if results.hand_landmarks:

                landmarks = results.hand_landmarks[0]

                draw_hand_landmarks(frame, landmarks)

                # ---------------------------------------------
                # A. Detect Pinch
                # ---------------------------------------------

                is_pinching, pinch_distance = detect_pinch(
                    landmarks
                )

                hand_status = "Normal"

                # ---------------------------------------------
                # B. Pinch has priority over letter recognition
                # ---------------------------------------------

                if is_pinching:

                    wrist_history.clear()

                    prediction_history.clear()

                    reset_letter_selection()

                    is_gesture_active, status_message = process_pinch(
                        True
                    )

                    hand_status = "PINCH"

                else:

                    # Release pinch timer
                    process_pinch(False)

                    # -----------------------------------------
                    # C. Detect Left Swipe
                    # -----------------------------------------

                    swipe_detected = detect_left_swipe(
                        landmarks
                    )

                    if swipe_detected:

                        delete_last()

                        prediction_history.clear()

                        reset_letter_selection()

                        current_prediction = "None"

                        status_message = "DELETE"

                        hand_status = "LEFT SWIPE"

                    else:

                        # -------------------------------------
                        # D. Extract 30 geometric features
                        # -------------------------------------

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

                            # Predict letter
                            prediction = model.predict(
                                feature_df
                            )[0]

                            current_prediction = prediction

                            # Stabilize prediction
                            stable_prediction = get_stable_prediction(
                                prediction
                            )

                            # Add letter after holding
                            status_message = process_letter_selection(
                                stable_prediction
                            )

                            if stable_prediction is not None:

                                hand_status = (
                                    f"Stable: {stable_prediction}"
                                )

                            else:

                                hand_status = "Stabilizing"

                        else:

                            prediction_history.clear()

                            reset_letter_selection()

                            status_message = "Feature extraction failed"

            # =================================================
            # NO HAND DETECTED
            # =================================================

            else:

                prediction_history.clear()

                wrist_history.clear()

                process_pinch(False)

                reset_letter_selection()

                current_prediction = "None"

                status_message = "Show your hand"

            # =================================================
            # DISPLAY INFORMATION
            # =================================================

            cv2.putText(
                frame,
                f"Prediction: {current_prediction}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status_message}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Text: {sentence}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Pinch: SPACE | Left Swipe: DELETE",
                (20, frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1
            )

            cv2.putText(
                frame,
                "C: Clear | Backspace: Delete | Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1
            )

            cv2.imshow(
                "Real-Time ASL Alphabet Translator",
                frame
            )

            # =================================================
            # KEYBOARD CONTROLS
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("c"):

                clear_text()

            elif key == 8:

                delete_last()

            elif key == 32:

                add_space()


finally:

    cap.release()

    cv2.destroyAllWindows()

    print("Final sentence:", sentence)

    print("Camera closed.")
