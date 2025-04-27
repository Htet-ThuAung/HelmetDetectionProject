import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from helmet_detector.config import config


PATH_TO_SAVED_MODEL = config.MODEL_PATH
PATH_TO_LABELS = config.LABELS_PATH
OUTPUT_VIDEO_PATH = config.OUTPUT_VIDEO_PATH

# Ensure output directory exists
output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
if not os.path.exists(output_dir):
    print(f"Creating output directory: {output_dir}")
    try:
        os.makedirs(output_dir)
    except Exception as e:
        print(f"Error creating directory: {e}")
        exit()

# Load the model
print("Loading model...")
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print("Model loaded.")

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Get original resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)  # Change codec here
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, config.FRAME_RATE, (frame_width, frame_height))

print(f"Saving video to: {OUTPUT_VIDEO_PATH}")
print("Starting webcam inference. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]

    # Run inference
    detections = detect_fn(input_tensor)

    # Extract and process detections
    num_detections = int(detections.pop("num_detections"))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    # Visualize results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=config.MAX_BOXES,
        min_score_thresh=config.MIN_SCORE_THRESH,
        agnostic_mode=False
    )

    # Show and save
    cv2.imshow("Helmet Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Webcam inference ended. Video saved to: {OUTPUT_VIDEO_PATH}")

# Show which script is running
import __main__
print(f"Running: {__main__.__file__}")
