
import csv
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "raw" / "val" / "images"
ANNOTATION_DIR = ROOT / "data" / "raw" / "val" / "annotations"

MODELS = {
    "original": ROOT / "models" / "exportedModel" / "saved_model",
}

# Optional checkpoint comparison models.
CHECKPOINT76_MODEL = (
    ROOT / "models" / "exportedModel_checkpoint76" / "saved_model"
)

CHECKPOINT78_MODEL = (
    ROOT / "models" / "exportedModel_checkpoint78" / "saved_model"
)

if (CHECKPOINT76_MODEL / "saved_model.pb").exists():
    MODELS["checkpoint76"] = CHECKPOINT76_MODEL

if (CHECKPOINT78_MODEL / "saved_model.pb").exists():
    MODELS["checkpoint78"] = CHECKPOINT78_MODEL


OUTPUT_DIR = ROOT / "test_output" / "model_evaluation_checkpoint_comparison"

CLASSES = {
    1: "helmet",
    2: "no_helmet",
}

SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


# --------------------------------------------------
# Dataset loading
# --------------------------------------------------

def load_ground_truth():
    """Load Pascal VOC annotations into pixel-coordinate boxes."""
    if not ANNOTATION_DIR.is_dir():
        raise FileNotFoundError(
            f"Annotation directory not found: {ANNOTATION_DIR}"
        )

    ground_truth = {}

    for xml_path in sorted(ANNOTATION_DIR.glob("*.xml")):
        root = ET.parse(xml_path).getroot()

        image_name = root.findtext("filename")

        if not image_name:
            raise ValueError(
                f"Missing filename in annotation: {xml_path}"
            )

        if image_name in ground_truth:
            raise ValueError(
                f"Duplicate annotation filename: {image_name}"
            )

        objects = []

        for obj in root.findall("object"):
            class_name = obj.findtext("name")

            class_id = next(
                (
                    identifier
                    for identifier, name in CLASSES.items()
                    if name == class_name
                ),
                None,
            )

            if class_id is None:
                raise ValueError(
                    f"Unknown class '{class_name}' in {xml_path}"
                )

            box = obj.find("bndbox")

            if box is None:
                raise ValueError(
                    f"Missing bounding box in {xml_path}"
                )

            coordinate_values = [
                box.findtext("xmin"),
                box.findtext("ymin"),
                box.findtext("xmax"),
                box.findtext("ymax"),
            ]

            if any(value is None for value in coordinate_values):
                raise ValueError(
                    f"Incomplete bounding box in {xml_path}"
                )

            xmin, ymin, xmax, ymax = [
                float(value) for value in coordinate_values
            ]

            # Convert Pascal VOC coordinates to zero-based coordinates.
            coordinates = [
                xmin - 1,
                ymin - 1,
                xmax,
                ymax,
            ]

            xmin, ymin, xmax, ymax = coordinates

            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f"Invalid bounding box in {xml_path}"
                )

            objects.append({
                "class_id": class_id,
                "box": coordinates,
            })

        ground_truth[image_name] = objects

    return ground_truth


# --------------------------------------------------
# Model loading and inference
# --------------------------------------------------

def load_model(model_path):
    """Load a TensorFlow SavedModel using its serving signature."""
    saved_model_file = model_path / "saved_model.pb"

    if not saved_model_file.exists():
        raise FileNotFoundError(
            f"SavedModel not found: {model_path}"
        )

    model = tf.saved_model.load(str(model_path))

    if "serving_default" not in model.signatures:
        raise ValueError(
            f"Missing serving_default signature: {model_path}"
        )

    return model.signatures["serving_default"]


def predict_image(infer, image_path):
    """Return model predictions in pixel coordinates."""
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(
        image_rgb[None, ...],
        dtype=tf.uint8,
    )

    outputs = infer(input_tensor)

    count = int(outputs["num_detections"][0].numpy())

    boxes = outputs["detection_boxes"][0, :count].numpy()
    scores = outputs["detection_scores"][0, :count].numpy()
    classes = (
        outputs["detection_classes"][0, :count]
        .numpy()
        .astype(int)
    )

    unknown_classes = set(classes) - set(CLASSES)

    if unknown_classes:
        raise ValueError(
            f"Model returned unknown class IDs: {unknown_classes}"
        )
    predictions = []

    for box, score, class_id in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = box

        predictions.append({
            "class_id": int(class_id),
            "score": float(score),
            "box": [
                float(xmin * width),
                float(ymin * height),
                float(xmax * width),
                float(ymax * height),
            ],
        })

    return predictions


# --------------------------------------------------
# IoU calculation
# --------------------------------------------------

def calculate_iou(box_a, box_b):
    """Calculate IoU for [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# --------------------------------------------------
# Average precision calculation
# --------------------------------------------------

def calculate_ap(recall, precision):
    """Calculate interpolated AP using 101 recall points."""
    if len(recall) == 0:
        return 0.0

    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated_precision = []

    for level in recall_levels:
        eligible = precision[recall >= level]

        interpolated_precision.append(
            float(np.max(eligible)) if len(eligible) else 0.0
        )

    return float(np.mean(interpolated_precision))


def evaluate_class(
    predictions_by_image,
    ground_truth,
    class_id,
    iou_threshold,
):
    """Calculate AP for one class at one IoU threshold."""
    gt_by_image = {}
    total_gt = 0

    for image_name, objects in ground_truth.items():
        boxes = [
            obj["box"]
            for obj in objects
            if obj["class_id"] == class_id
        ]

        gt_by_image[image_name] = {
            "boxes": boxes,
            "matched": [False] * len(boxes),
        }

        total_gt += len(boxes)

    if total_gt == 0:
        return 0.0

    all_predictions = []

    for image_name, predictions in predictions_by_image.items():
        for prediction in predictions:
            if prediction["class_id"] == class_id:
                all_predictions.append({
                    "image": image_name,
                    "score": prediction["score"],
                    "box": prediction["box"],
                })

    all_predictions.sort(
        key=lambda prediction: prediction["score"],
        reverse=True,
    )

    true_positives = []
    false_positives = []

    for prediction in all_predictions:
        image_name = prediction["image"]
        gt_data = gt_by_image[image_name]

        best_iou = 0.0
        best_index = -1

        for index, gt_box in enumerate(gt_data["boxes"]):
            if gt_data["matched"][index]:
                continue

            iou = calculate_iou(
                prediction["box"],
                gt_box,
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_iou >= iou_threshold and best_index >= 0:
            gt_data["matched"][best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    if not true_positives:
        return 0.0

    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)

    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)

    return calculate_ap(recall, precision)


# --------------------------------------------------
# Precision, recall and F1 calculation
# --------------------------------------------------

def calculate_threshold_metrics(
    predictions_by_image,
    ground_truth,
    class_id,
):
    """Calculate precision, recall and F1 at fixed thresholds."""
    tp = 0
    fp = 0
    fn = 0

    for image_name, objects in ground_truth.items():
        gt_boxes = [
            obj["box"]
            for obj in objects
            if obj["class_id"] == class_id
        ]

        predictions = [
            prediction
            for prediction in predictions_by_image.get(image_name, [])
            if (
                prediction["class_id"] == class_id
                and prediction["score"] >= SCORE_THRESHOLD
            )
        ]

        predictions.sort(
            key=lambda prediction: prediction["score"],
            reverse=True,
        )

        matched = [False] * len(gt_boxes)

        for prediction in predictions:
            best_iou = 0.0
            best_index = -1

            for index, gt_box in enumerate(gt_boxes):
                if matched[index]:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    gt_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_iou >= IOU_THRESHOLD and best_index >= 0:
                matched[best_index] = True
                tp += 1
            else:
                fp += 1

        fn += matched.count(False)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------
# Model evaluation
# --------------------------------------------------

def evaluate_model(model_name, model_path, ground_truth, image_paths):
    """Evaluate one model on the validation dataset."""
    print(f"\nLoading {model_name}...")

    infer = load_model(model_path)

    # Warm up the model before measuring inference time.
    print(f"Warming up {model_name}...")

    for _ in range(3):
        predict_image(infer, image_paths[0])

    predictions_by_image = {}
    inference_times = []

    # End-to-end timing includes image loading, preprocessing,
    # TensorFlow inference, and prediction extraction.
    for index, image_path in enumerate(image_paths, start=1):
        start = time.perf_counter()

        predictions = predict_image(infer, image_path)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        inference_times.append(elapsed_ms)
        predictions_by_image[image_path.name] = predictions

        if index % 25 == 0 or index == len(image_paths):
            print(
                f"{model_name}: "
                f"{index}/{len(image_paths)} images"
            )

    class_results = {}
    all_ap_values = []

    for class_id, class_name in CLASSES.items():
        ap_by_threshold = {}

        for threshold in IOU_THRESHOLDS:
            ap = evaluate_class(
                predictions_by_image,
                ground_truth,
                class_id,
                float(threshold),
            )

            ap_by_threshold[f"AP{threshold:.2f}"] = ap
            all_ap_values.append(ap)

        ap50 = ap_by_threshold["AP0.50"]

        map_50_95 = float(
            np.mean(list(ap_by_threshold.values()))
        )

        threshold_metrics = calculate_threshold_metrics(
            predictions_by_image,
            ground_truth,
            class_id,
        )

        class_results[class_name] = {
            **threshold_metrics,
            "AP50": ap50,
            "AP50_95": map_50_95,
            **ap_by_threshold,
        }

    # Macro-average across classes and IoU thresholds.
    overall_map_50_95 = float(np.mean(all_ap_values))

    overall_map50 = float(
        np.mean([
            class_results[name]["AP50"]
            for name in CLASSES.values()
        ])
    )

    total_tp = sum(
        result["TP"] for result in class_results.values()
    )

    total_fp = sum(
        result["FP"] for result in class_results.values()
    )

    total_fn = sum(
        result["FN"] for result in class_results.values()
    )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "model": model_name,
        "images_evaluated": len(image_paths),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": overall_map50,
        "mAP50_95": overall_map_50_95,
        "mean_inference_ms": float(np.mean(inference_times)),
        "median_inference_ms": float(np.median(inference_times)),
        "classes": class_results,
        "predictions": predictions_by_image,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading validation annotations...")
    ground_truth = load_ground_truth()

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        raise ValueError(
            f"No validation images found in {IMAGE_DIR}"
        )

    annotated_names = set(ground_truth)
    image_names = {path.name for path in image_paths}

    if annotated_names != image_names:
        raise ValueError(
            "Image and annotation filenames do not match. "
            f"Missing annotations: "
            f"{sorted(image_names - annotated_names)[:5]}; "
            f"Missing images: "
            f"{sorted(annotated_names - image_names)[:5]}"
        )

    print(f"Validation images: {len(image_paths)}")
    print(f"Annotated images: {len(ground_truth)}")

    print("\nModels selected for evaluation:")

    for model_name, model_path in MODELS.items():
        print(f"- {model_name}: {model_path}")

    results = []

    for model_name, model_path in MODELS.items():
        result = evaluate_model(
            model_name,
            model_path,
            ground_truth,
            image_paths,
        )

        results.append(result)

    # --------------------------------------------------
    # Summary CSV
    # --------------------------------------------------

    summary_path = OUTPUT_DIR / "evaluation_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "images_evaluated",
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50_95",
            "mean_inference_ms",
            "median_inference_ms",
        ])

        for result in results:
            writer.writerow([
                result["model"],
                result["images_evaluated"],
                result["precision"],
                result["recall"],
                result["f1"],
                result["mAP50"],
                result["mAP50_95"],
                result["mean_inference_ms"],
                result["median_inference_ms"],
            ])

    # --------------------------------------------------
    # Per-class metrics JSON
    # --------------------------------------------------

    class_path = OUTPUT_DIR / "per_class_metrics.json"

    class_data = {
        result["model"]: result["classes"]
        for result in results
    }

    with class_path.open("w", encoding="utf-8") as file:
        json.dump(class_data, file, indent=4)

    # --------------------------------------------------
    # Raw prediction CSV
    # --------------------------------------------------

    prediction_path = OUTPUT_DIR / "predictions.csv"

    with prediction_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "model",
            "image",
            "class_id",
            "class_name",
            "confidence",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ])

        for result in results:
            for image_name, predictions in result["predictions"].items():
                for prediction in predictions:
                    writer.writerow([
                        result["model"],
                        image_name,
                        prediction["class_id"],
                        CLASSES.get(
                            prediction["class_id"],
                            "unknown",
                        ),
                        prediction["score"],
                        *prediction["box"],
                    ])

    # --------------------------------------------------
    # Terminal summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("VALIDATION EVALUATION RESULTS")
    print("=" * 65)

    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"Images: {result['images_evaluated']}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1-score: {result['f1']:.4f}")
        print(f"mAP@0.5: {result['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {result['mAP50_95']:.4f}")
        print(
            f"Mean inference: "
            f"{result['mean_inference_ms']:.2f} ms"
        )
        print(
            f"Median inference: "
            f"{result['median_inference_ms']:.2f} ms"
        )

        for class_name, metrics in result["classes"].items():
            print(
                f"  {class_name}: "
                f"AP50={metrics['AP50']:.4f}, "
                f"AP50:95={metrics['AP50_95']:.4f}"
            )

    print("\nReports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
