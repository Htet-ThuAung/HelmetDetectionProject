
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf

from helmet_detector.object_detection.utils import label_map_util
from helmet_detector.object_detection.utils import visualization_utils as viz_utils
from helmet_detector.config import config


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SavedModel not found: {model_path}")

    print("Loading model...")
    detect_fn = tf.saved_model.load(model_path)
    print("Model loaded.")

    return detect_fn


def run_inference(
    image_path,
    model,
    label_map_path,
    score_thresh=0.5,
    output_path=None
):
    # Validate input files and threshold
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not os.path.isfile(label_map_path):
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    if not 0.0 <= score_thresh <= 1.0:
        raise ValueError("Score threshold must be between 0.0 and 1.0.")

    # Read image
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Convert BGR to RGB for inference
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(
        image_rgb,
        dtype=tf.uint8
    )[tf.newaxis, ...]

    # Load label map
    category_index = label_map_util.create_category_index_from_labelmap(
        label_map_path,
        use_display_name=True
    )

    # Run inference
    detections = model(input_tensor)

    num_detections = int(detections.pop("num_detections"))

    detections = {
        key: value[0, :num_detections].numpy()
        for key, value in detections.items()
    }

    detections["detection_classes"] = (
        detections["detection_classes"].astype(np.int64)
    )

    # Draw detections on RGB image
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_rgb,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=config.MAX_BOXES,
        min_score_thresh=score_thresh,
        agnostic_mode=False
    )

    # Convert annotated image back to BGR
    annotated_image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    # Save output image if requested
    if output_path:
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if not cv2.imwrite(output_path, annotated_image):
            raise IOError(f"Unable to save output image: {output_path}")

        print(f"Saved output to: {output_path}")

    # Display result
    try:
        cv2.imshow("Helmet Detection", annotated_image)
        print("Press any key to close the image window.")
        cv2.waitKey(0)

    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Helmet Detection Image Inference"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Optional path to save annotated image"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=config.MODEL_PATH,
        help="Path to exported SavedModel"
    )

    parser.add_argument(
        "--label_map",
        type=str,
        default=config.LABEL_MAP_PATH,
        help="Path to label map"
    )

    parser.add_argument(
        "--score_thresh",
        type=float,
        default=config.MIN_SCORE_THRESH,
        help="Minimum confidence threshold"
    )

    args = parser.parse_args()

    model = load_model(args.model_dir)

    run_inference(
        image_path=args.image_path,
        model=model,
        label_map_path=args.label_map,
        score_thresh=args.score_thresh,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
