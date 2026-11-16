
import cv2
import mediapipe as mp
import csv
import handlandmark

from pathlib import Path
from tqdm import tqdm


# ==========================================
# 1. PATHS
# ==========================================

DATASET_PATH = Path(
    r"E:\Ai&Machine learning\ASL_translator\data\asl_alphabet_train\asl_alphabet_train"
)

PROJECT_PATH = Path(__file__).resolve().parent

OUTPUT_PATH = PROJECT_PATH / "data" / "processed" / "asl_features.csv"
FAILED_PATH = PROJECT_PATH / "data" / "processed" / "failed_images.csv"

MODEL_PATH = PROJECT_PATH / "models" / "hand_landmarker.task"


# ==========================================
# 2. CONFIGURATION
# ==========================================

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

EXPECTED_DISTANCE_FEATURES = 15
EXPECTED_ANGLE_FEATURES = 10
EXPECTED_STRAIGHTNESS_FEATURES = 5
EXPECTED_TOTAL_FEATURES = 30


# ==========================================
# 3. CHECK PATHS AND FUNCTIONS
# ==========================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH}"
    )

required_functions = [
    "extract_features",
    "extract_distance_features",
    "extract_angle_features",
    "extract_straightness_features"
]

for function_name in required_functions:
    if not hasattr(handlandmark, function_name):
        raise AttributeError(
            f"Missing function in handlandmark.py: {function_name}"
        )


# ==========================================
# 4. MEDIAPIPE INITIALIZATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)


# ==========================================
# 5. GET DATASET IMAGES
# ==========================================

def get_images():

    images = []

    for label in LABELS:

        class_folder = DATASET_PATH / label

        if not class_folder.exists():
            print(f"Warning: Missing class folder: {label}")
            continue

        class_images = list(class_folder.glob("*.jpg"))

        for image_path in class_images:
            images.append((image_path, label))

    return images


# ==========================================
# 6. VALIDATE FEATURE EXTRACTION
# ==========================================

def validate_features(landmarks):

    # Extract each feature group separately
    distance_features = (
        handlandmark.extract_distance_features(landmarks)
    )

    angle_features = (
        handlandmark.extract_angle_features(landmarks)
    )

    straightness_features = (
        handlandmark.extract_straightness_features(landmarks)
    )

    # Check each group
    if len(distance_features) != EXPECTED_DISTANCE_FEATURES:
        raise ValueError(
            f"Expected 15 distance features, "
            f"got {len(distance_features)}"
        )

    if len(angle_features) != EXPECTED_ANGLE_FEATURES:
        raise ValueError(
            f"Expected 10 angle features, "
            f"got {len(angle_features)}"
        )

    if len(straightness_features) != EXPECTED_STRAIGHTNESS_FEATURES:
        raise ValueError(
            f"Expected 5 straightness features, "
            f"got {len(straightness_features)}"
        )

    # Extract using the same function used by live prediction
    all_features = handlandmark.extract_features(landmarks)

    # Independently combine the three groups
    expected_features = {}

    expected_features.update(distance_features)
    expected_features.update(angle_features)
    expected_features.update(straightness_features)

    # Verify total count
    if len(all_features) != EXPECTED_TOTAL_FEATURES:
        raise ValueError(
            f"Expected 30 total features, "
            f"got {len(all_features)}"
        )

    # Verify that extract_features returns all expected keys
    if set(all_features.keys()) != set(expected_features.keys()):
        raise ValueError(
            "Feature names do not match the three feature groups."
        )

    # Verify values are consistent
    for key in expected_features:

        if abs(
            all_features[key] - expected_features[key]
        ) > 1e-8:

            raise ValueError(
                f"Feature mismatch: {key}"
            )

    return all_features


# ==========================================
# 7. PROCESS ONE IMAGE
# ==========================================

def process_image(image_path, landmarker):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read image")

    # BGR -> RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    # Detect hand landmarks
    results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        raise ValueError("No hand detected")

    landmarks = results.hand_landmarks[0]

    # Extract and validate all 30 features
    features = validate_features(landmarks)

    return features


# ==========================================
# 8. MAIN PIPELINE
# ==========================================

def main():

    print("=" * 50)
    print("ASL DATASET FEATURE EXTRACTION")
    print("=" * 50)

    images = get_images()

    print(f"Total images found: {len(images)}")

    if not images:
        raise RuntimeError(
            "No images found. Check DATASET_PATH and class folders."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    FAILED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    successful_images = 0
    failed_images = 0

    feature_columns = None

    # Initialize MediaPipe
    with HandLandmarker.create_from_options(options) as landmarker:

        # Open CSV files once
        with open(
            OUTPUT_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as output_file, open(
            FAILED_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as failed_file:

            writer = csv.writer(output_file)
            failed_writer = csv.writer(failed_file)

            # Failed CSV header
            failed_writer.writerow([
                "label",
                "image_path",
                "error"
            ])

            # Process images
            for image_path, label in tqdm(
                images,
                desc="Extracting features"
            ):

                try:

                    features = process_image(
                        image_path,
                        landmarker
                    )

                    # Initialize CSV header from first valid image
                    if feature_columns is None:

                        feature_columns = list(
                            features.keys()
                        )

                        if len(feature_columns) != 30:
                            raise ValueError(
                                "CSV initialization failed: "
                                "Expected exactly 30 features."
                            )

                        writer.writerow(
                            ["label", "image_path"] + feature_columns
                        )

                        print("\nFeature groups verified:")
                        print("Distance features: 15")
                        print("Angle features: 10")
                        print("Straightness features: 5")
                        print("Total features: 30")

                        print("\nCSV columns:")
                        print(feature_columns)

                    # Ensure every image has the same feature names
                    if list(features.keys()) != feature_columns:
                        raise ValueError(
                            "Feature columns/order changed."
                        )

                    # Build row in the exact header order
                    row = [
                        label,
                        str(image_path)
                    ]

                    row.extend(
                        features[column]
                        for column in feature_columns
                    )

                    # Final row validation
                    if len(row) != 32:
                        raise ValueError(
                            f"Expected 32 CSV values, got {len(row)}"
                        )

                    writer.writerow(row)

                    successful_images += 1

                except Exception as error:

                    failed_writer.writerow([
                        label,
                        str(image_path),
                        str(error)
                    ])

                    failed_images += 1

    # ======================================
    # 9. FINAL RESULTS
    # ======================================

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    print(f"Total images: {len(images)}")
    print(f"Successful images: {successful_images}")
    print(f"Failed images: {failed_images}")

    if feature_columns is None:
        raise RuntimeError(
            "No valid images were processed. CSV was not created."
        )

    print(f"Number of features: {len(feature_columns)}")
    print(f"Number of CSV columns: {len(feature_columns) + 2}")

    print(f"\nSaved CSV: {OUTPUT_PATH}")
    print(f"Failed images CSV: {FAILED_PATH}")

    if len(feature_columns) != 30:
        raise RuntimeError(
            "Final validation failed: CSV must contain 30 features."
        )

    print("\nSUCCESS: Dataset contains exactly 30 features per image!")


# ==========================================
# 10. RUN
# ==========================================

if __name__ == "__main__":
    main()
