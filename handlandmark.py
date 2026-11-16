
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_PATH = Path("models/hand_landmarker.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Hand Landmarker model not found: {MODEL_PATH}"
    )


# ==================================================
# 2. MEDIAPIPE TASKS API
# ==================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)


# ==================================================
# 3. HAND CONNECTIONS
# ==================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ==================================================
# 4. DRAW HAND LANDMARKS
# ==================================================

def draw_landmarks(frame, landmarks):

    h, w, _ = frame.shape

    points = []

    for landmark in landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

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

    # Draw landmark points and IDs
    for landmark_id, (x, y) in enumerate(points):

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            str(landmark_id),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )


# ==================================================
# 5. EUCLIDEAN DISTANCE
# ==================================================

def calculate_distance(point1, point2):

    dx = point1.x - point2.x
    dy = point1.y - point2.y
    dz = point1.z - point2.z

    distance = np.sqrt(
        dx**2 +
        dy**2 +
        dz**2
    )

    return distance


# ==================================================
# 6. DISTANCE FEATURES (15 FEATURES)
# ==================================================

def extract_distance_features(landmarks):

    fingertips = [4, 8, 12, 16, 20]

    # Palm reference distance
    palm_size = calculate_distance(
        landmarks[0],
        landmarks[9]
    )

    palm_size = max(palm_size, 1e-6)

    features = {}

    # A. Fingertip-to-Fingertip distances
    for i in range(len(fingertips)):

        for j in range(i + 1, len(fingertips)):

            id1 = fingertips[i]
            id2 = fingertips[j]

            distance = calculate_distance(
                landmarks[id1],
                landmarks[id2]
            )

            normalized_distance = distance / palm_size

            feature_name = f"tip_{id1}_to_tip_{id2}"

            features[feature_name] = normalized_distance

    # B. Wrist-to-Fingertip distances
    wrist_id = 0

    for tip_id in fingertips:

        distance = calculate_distance(
            landmarks[wrist_id],
            landmarks[tip_id]
        )

        normalized_distance = distance / palm_size

        feature_name = f"wrist_to_tip_{tip_id}"

        features[feature_name] = normalized_distance

    return features


# ==================================================
# 7. CALCULATE JOINT ANGLE
# ==================================================

def calculate_angle(point_a, point_b, point_c):

    a = np.array([
        point_a.x,
        point_a.y,
        point_a.z
    ])

    b = np.array([
        point_b.x,
        point_b.y,
        point_b.z
    ])

    c = np.array([
        point_c.x,
        point_c.y,
        point_c.z
    ])

    # Vectors from middle point
    vector_ba = a - b
    vector_bc = c - b

    # Dot product
    dot_product = np.dot(
        vector_ba,
        vector_bc
    )

    # Vector magnitudes
    magnitude_ba = np.linalg.norm(vector_ba)
    magnitude_bc = np.linalg.norm(vector_bc)

    denominator = magnitude_ba * magnitude_bc

    if denominator < 1e-8:
        return 0.0

    cosine_angle = dot_product / denominator

    # Numerical stability
    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    # Radians to degrees
    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return angle


# ==================================================
# 8. ANGLE FEATURES (10 FEATURES)
# ==================================================

def extract_angle_features(landmarks):

    finger_joints = {

        "thumb": [
            (1, 2, 3),
            (2, 3, 4)
        ],

        "index": [
            (5, 6, 7),
            (6, 7, 8)
        ],

        "middle": [
            (9, 10, 11),
            (10, 11, 12)
        ],

        "ring": [
            (13, 14, 15),
            (14, 15, 16)
        ],

        "pinky": [
            (17, 18, 19),
            (18, 19, 20)
        ]
    }

    features = {}

    for finger_name, joints in finger_joints.items():

        for angle_index, (a, b, c) in enumerate(
            joints,
            start=1
        ):

            angle = calculate_angle(
                landmarks[a],
                landmarks[b],
                landmarks[c]
            )

            feature_name = (
                f"{finger_name}_angle_{angle_index}"
            )

            features[feature_name] = angle

    return features


# ==================================================
# 9. FINGER STRAIGHTNESS
# ==================================================

def calculate_straightness(landmarks, ids):

    # Direct distance between first and last points
    direct_distance = calculate_distance(
        landmarks[ids[0]],
        landmarks[ids[-1]]
    )

    # Sum of all segment distances
    path_distance = 0

    for i in range(len(ids) - 1):

        path_distance += calculate_distance(
            landmarks[ids[i]],
            landmarks[ids[i + 1]]
        )

    if path_distance < 1e-8:
        return 0.0

    return direct_distance / path_distance


# ==================================================
# 10. STRAIGHTNESS FEATURES (5 FEATURES)
# ==================================================

def extract_straightness_features(landmarks):

    finger_segments = {

        "thumb": [1, 2, 3, 4],

        "index": [5, 6, 7, 8],

        "middle": [9, 10, 11, 12],

        "ring": [13, 14, 15, 16],

        "pinky": [17, 18, 19, 20]
    }

    features = {}

    for finger_name, ids in finger_segments.items():

        straightness = calculate_straightness(
            landmarks,
            ids
        )

        features[
            f"{finger_name}_straightness"
        ] = straightness

    return features


# ==================================================
# 11. COMBINE ALL GEOMETRIC FEATURES (30)
# ==================================================

def extract_features(landmarks):

    # 15 Distance Features
    distance_features = extract_distance_features(
        landmarks
    )

    # 10 Joint Angle Features
    angle_features = extract_angle_features(
        landmarks
    )

    # 5 Straightness Features
    straightness_features = extract_straightness_features(
        landmarks
    )

    # Combine everything
    features = {}

    features.update(distance_features)
    features.update(angle_features)
    features.update(straightness_features)

    return features


# ==================================================
# 12. PRINT ALL FEATURES
# ==================================================

def print_features(features):

    print("\n========== ASL GEOMETRIC FEATURES ==========")

    print("\n--- DISTANCE FEATURES ---")

    for name, value in features.items():

        if name.startswith("tip_") or name.startswith("wrist_"):

            print(f"{name:30s}: {value:.4f}")

    print("\n--- ANGLE FEATURES ---")

    for name, value in features.items():

        if "_angle_" in name:

            print(f"{name:30s}: {value:.2f} degrees")

    print("\n--- STRAIGHTNESS FEATURES ---")

    for name, value in features.items():

        if "_straightness" in name:

            print(f"{name:30s}: {value:.4f}")

    print("\n--------------------------------------------")
    print(f"Total Features: {len(features)}")
    print("============================================")


# ==================================================
# 13. MAIN WEBCAM PIPELINE
# ==================================================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Cannot open webcam"
        )

    last_timestamp_ms = -1

    with HandLandmarker.create_from_options(options) as landmarker:

        latest_features = None

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:

                print("Cannot read webcam frame")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Create MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Monotonic timestamp
            timestamp_ms = max(
                time.monotonic_ns() // 1_000_000,
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # Detect hand
            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            latest_features = None

            # --------------------------------------
            # HAND DETECTION
            # --------------------------------------

            if results.hand_landmarks:

                landmarks = results.hand_landmarks[0]

                # Draw skeleton
                draw_landmarks(
                    frame,
                    landmarks
                )

                # Extract 30 features
                latest_features = extract_features(
                    landmarks
                )

                cv2.putText(
                    frame,
                    "Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Geometric Features: 30",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # Show webcam
            cv2.imshow(
                "ASL - Geometric Feature Extraction",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # --------------------------------------
            # PRESS P TO PRINT FEATURES
            # --------------------------------------

            if key == ord('p'):

                if latest_features is not None:

                    print_features(
                        latest_features
                    )

                else:

                    print("No hand detected!")

            # --------------------------------------
            # PRESS Q TO QUIT
            # --------------------------------------

            elif key == ord('q'):

                break

    # ==================================================
    # 14. RELEASE RESOURCES
    # ==================================================

    cap.release()
    cv2.destroyAllWindows()

    print("\nWebcam closed successfully.")


# ==================================================
# 15. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()
