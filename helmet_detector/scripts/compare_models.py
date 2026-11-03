
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "helmet_detector" / "models"

ORIGINAL_MODEL_PATH = MODELS_DIR / "exportedModel" / "saved_model"
CHECKPOINT76_MODEL_PATH = (
    MODELS_DIR / "exportedModel_checkpoint76" / "saved_model"
)

IMAGE_DIR = PROJECT_ROOT / "helmet_detector" / "test_output" / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "helmet_detector" / "test_output" / "model_comparison"

SCORE_THRESHOLD = 0.5
MAX_IMAGES = 10


def load_model(model_path):
    """Load a TensorFlow SavedModel and its serving signature."""
    model = tf.saved_model.load(str(model_path))
    return model.signatures["serving_default"]


def run_inference(infer, image_rgb):
    """Run inference and measure execution time."""
    input_tensor = tf.convert_to_tensor(image_rgb[None, ...], dtype=tf.uint8)

    start_time = time.perf_counter()
    outputs = infer(input_tensor)
    elapsed = time.perf_counter() - start_time

    num_detections = int(outputs["num_detections"][0].numpy())

    boxes = outputs["detection_boxes"][0, :num_detections].numpy()
    scores = outputs["detection_scores"][0, :num_detections].numpy()
    classes = outputs["detection_classes"][0, :num_detections].numpy().astype(int)

    valid = scores >= SCORE_THRESHOLD

    return {
        "boxes": boxes[valid],
        "scores": scores[valid],
        "classes": classes[valid],
        "inference_time_ms": elapsed * 1000,
    }


def draw_detections(image, detections, title):
    """Draw predicted bounding boxes on an image."""
    height, width = image.shape[:2]
    output = image.copy()

    for box, score, class_id in zip(
        detections["boxes"],
        detections["scores"],
        detections["classes"],
    ):
        ymin, xmin, ymax, xmax = box

        x1 = int(xmin * width)
        y1 = int(ymin * height)
        x2 = int(xmax * width)
        y2 = int(ymax * height)

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        output,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original model...")
    original_model = load_model(ORIGINAL_MODEL_PATH)

    print("Loading checkpoint 76...")
    checkpoint76_model = load_model(CHECKPOINT76_MODEL_PATH)

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:MAX_IMAGES]

    if not image_paths:
        raise FileNotFoundError(f"No test images found in {IMAGE_DIR}")

    report_rows = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        original = run_inference(original_model, image_rgb)
        checkpoint76 = run_inference(checkpoint76_model, image_rgb)

        print(f"\nImage: {image_path.name}")
        print(
            f"Original: {len(original['scores'])} detections, "
            f"{original['inference_time_ms']:.2f} ms"
        )
        print(
            f"Checkpoint 76: {len(checkpoint76['scores'])} detections, "
            f"{checkpoint76['inference_time_ms']:.2f} ms"
        )

        report_rows.append({
            "image": image_path.name,
            "original_detections": len(original["scores"]),
            "checkpoint76_detections": len(checkpoint76["scores"]),
            "original_avg_confidence": (
                float(np.mean(original["scores"])) if len(original["scores"]) else 0
            ),
            "checkpoint76_avg_confidence": (
                float(np.mean(checkpoint76["scores"])) if len(checkpoint76["scores"]) else 0
            ),
            "original_inference_ms": original["inference_time_ms"],
            "checkpoint76_inference_ms": checkpoint76["inference_time_ms"],
        })

        original_image = draw_detections(
            image_bgr, original, "Original model"
        )
        checkpoint76_image = draw_detections(
            image_bgr, checkpoint76, "Checkpoint 76"
        )

        comparison = np.hstack((original_image, checkpoint76_image))

        output_path = OUTPUT_DIR / f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(str(output_path), comparison)

    if not report_rows:
        raise RuntimeError("No images were successfully processed.")

    csv_path = OUTPUT_DIR / "comparison_report.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nComparison report saved: {csv_path}")
    print(f"Annotated images saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
