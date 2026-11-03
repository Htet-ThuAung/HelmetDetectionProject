
import os

import cv2
import numpy as np
import tensorflow as tf

from helmet_detector.object_detection.utils import label_map_util
from helmet_detector.object_detection.utils import visualization_utils as viz_utils
from helmet_detector.config import config


PATH_TO_SAVED_MODEL = config.MODEL_PATH
PATH_TO_LABELS = config.LABEL_MAP_PATH
OUTPUT_VIDEO_PATH = config.OUTPUT_VIDEO_PATH


def main():
    # Validate required paths
    if not os.path.exists(PATH_TO_SAVED_MODEL):
        raise FileNotFoundError(
            f"SavedModel not found: {PATH_TO_SAVED_MODEL}"
        )

    if not os.path.exists(PATH_TO_LABELS):
        raise FileNotFoundError(
            f"Label map not found: {PATH_TO_LABELS}"
        )

    # Create output directory
    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model loaded.")

    # Load label map
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS,
        use_display_name=True
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Unable to access the webcam.")

    out = None

    try:
        # Get webcam resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_width <= 0 or frame_height <= 0:
            raise RuntimeError("Unable to determine webcam resolution.")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)

        out = cv2.VideoWriter(
            OUTPUT_VIDEO_PATH,
            fourcc,
            config.FRAME_RATE,
            (frame_width, frame_height)
        )

        if not out.isOpened():
            raise RuntimeError(
                f"Unable to initialize video writer: {OUTPUT_VIDEO_PATH}"
            )

        print(f"Saving video to: {OUTPUT_VIDEO_PATH}")
        print("Starting webcam inference. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Convert BGR to RGB for model inference
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            input_tensor = tf.convert_to_tensor(
                image_rgb,
                dtype=tf.uint8
            )[tf.newaxis, ...]

            # Run inference
            detections = detect_fn(input_tensor)

            # Process detections
            num_detections = int(detections.pop("num_detections"))

            detections = {
                key: value[0, :num_detections].numpy()
                for key, value in detections.items()
            }

            detections["detection_classes"] = (
                detections["detection_classes"].astype(np.int64)
            )

            # Draw detections on the RGB image
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_rgb,
                detections["detection_boxes"],
                detections["detection_classes"],
                detections["detection_scores"],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=config.MAX_BOXES,
                min_score_thresh=config.MIN_SCORE_THRESH,
                agnostic_mode=False
            )

            # Convert annotated image back to BGR for OpenCV
            annotated_frame = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR
            )

            # Display and save annotated frame
            cv2.imshow("Helmet Detection", annotated_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Always release resources
        cap.release()

        if out is not None:
            out.release()

        cv2.destroyAllWindows()

        print("Webcam resources released.")

    print(f"Webcam inference ended. Video saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
