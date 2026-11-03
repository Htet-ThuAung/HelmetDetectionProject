
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAVED_MODEL_DIR = (
    PROJECT_ROOT
    / "helmet_detector"
    / "models"
    / "exportedModel_checkpoint78"
    / "saved_model"
)

CLASS_NAMES = {
    1: "helmet",
    2: "no_helmet",
}

REQUIRED_OUTPUTS = {
    "detection_boxes",
    "detection_scores",
    "detection_classes",
    "num_detections",
}


def main(image_path, threshold):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not (SAVED_MODEL_DIR / "saved_model.pb").is_file():
        raise FileNotFoundError(f"SavedModel not found: {SAVED_MODEL_DIR}")

    print("[INFO] Loading exported model...")
    detect_fn = tf.saved_model.load(str(SAVED_MODEL_DIR))
    infer = detect_fn.signatures["serving_default"]
    print("[SUCCESS] SavedModel loaded.")

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    input_tensor = tf.convert_to_tensor(
        image_np, dtype=tf.uint8
    )[tf.newaxis, ...]

    print("[INFO] Running inference...")
    detections = infer(input_tensor)

    missing_outputs = REQUIRED_OUTPUTS - set(detections.keys())
    if missing_outputs:
        raise RuntimeError(
            f"Missing expected detection outputs: {missing_outputs}"
        )

    num_detections = int(detections["num_detections"][0].numpy())

    boxes = detections["detection_boxes"][0].numpy()[:num_detections]
    scores = detections["detection_scores"][0].numpy()[:num_detections]
    classes = detections["detection_classes"][0].numpy()[:num_detections]

    valid_indices = np.where(scores >= threshold)[0]

    print("\n[SUCCESS] Inference completed.")
    print(f"Candidate detections: {num_detections}")
    print(f"Confidence threshold: {threshold:.2f}")
    print(f"Valid detections: {len(valid_indices)}")

    if len(valid_indices) == 0:
        print("[INFO] No detections met the confidence threshold.")
        return

    for index in valid_indices:
        class_id = int(classes[index])
        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")

        print(
            f"Detection {index + 1}: "
            f"class={class_name}, "
            f"confidence={scores[index]:.4f}, "
            f"box={np.round(boxes[index], 4)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify an exported helmet detection model."
    )

    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (default: 0.5)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")

    main(args.image_path, args.threshold)
