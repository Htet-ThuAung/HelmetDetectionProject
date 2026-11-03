
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "helmet_detector"
    / "models"
    / "exportedModel_checkpoint76"
    / "saved_model"
)

LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "helmet_detector"
    / "data"
    / "processed"
    / "label_map"
    / "label_map.pbtxt"
)

OUTPUT_DIR = PROJECT_ROOT / "helmet_detector" / "test_output"


def load_labels(path):
    from helmet_detector.object_detection.utils import label_map_util

    label_map = label_map_util.load_labelmap(str(path))
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=2,
        use_display_name=True,
    )

    return {
        item["id"]: item["name"]
        for item in categories
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m helmet_detector.scripts.visualize_export <image_path>"
        )

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    labels = load_labels(LABEL_MAP_PATH)

    print("[INFO] Loading SavedModel...")
    model = tf.saved_model.load(str(MODEL_PATH))
    infer = model.signatures["serving_default"]

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(image_rgb[None, ...], dtype=tf.uint8)
    detections = infer(input_tensor)

    count = int(detections["num_detections"][0].numpy())

    boxes = detections["detection_boxes"][0].numpy()[:count]
    scores = detections["detection_scores"][0].numpy()[:count]
    classes = detections["detection_classes"][0].numpy()[:count]

    threshold = 0.5
    height, width = image_bgr.shape[:2]

    drawn = 0

    for box, score, class_id in zip(boxes, scores, classes):
        if score < threshold:
            continue

        y_min, x_min, y_max, x_max = box

        x1 = int(x_min * width)
        y1 = int(y_min * height)
        x2 = int(x_max * width)
        y2 = int(y_max * height)

        class_name = labels.get(int(class_id), f"class_{int(class_id)}")
        text = f"{class_name}: {score:.2f}"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        drawn += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_checkpoint76.jpg"

    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError("Failed to save visualization.")

    print(f"[SUCCESS] Detections drawn: {drawn}")
    print(f"[SUCCESS] Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
