
import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from helmet_detector.config import config


def load_model(model_path):
    print("Loading model...")
    detect_fn = tf.saved_model.load(model_path)
    print("Model loaded.")
    return detect_fn

def run_inference(image_path, model, label_map_path, score_thresh=0.4, output_path=None):
    image_np = cv2.imread(image_path)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    detections = model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=score_thresh,
        agnostic_mode=False,
    )

    cv2.imshow("Helmet Detection", image_np)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image_np)
        print(f"Saved output to: {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(dir(config))
    parser = argparse.ArgumentParser(description="Helmet Detection Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, help="Optional path to save output image")
    parser.add_argument("--model_dir", type=str, default=config.MODEL_PATH)
    parser.add_argument("--label_map", type=str, default=config.LABELS_PATH)
    parser.add_argument("--score_thresh", type=float, default=config.MIN_SCORE_THRESH, help="Minimum score threshold for displaying boxes")

    args = parser.parse_args()

    model = load_model(args.model_dir)
    run_inference(args.image_path, model, args.label_map, args.score_thresh, args.output_path)

