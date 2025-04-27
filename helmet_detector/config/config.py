import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "train", "images")
RAW_ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "raw", "train", "annotations")

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "val", "images")
VAL_ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "raw", "val", "annotations")

# Preprocessed output
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "train")
PROCESSED_IMAGES_DIR = os.path.join(PROCESSED_DATA_DIR, "images")
PROCESSED_ANNOTATIONS_DIR = os.path.join(PROCESSED_DATA_DIR, "annotations")

# TFRecord paths (relative to BASE_DIR)
TFRECORD_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_TFRECORD_PATH = os.path.join(TFRECORD_DIR, "train.record")

# Path for train annotation CSV (relative to BASE_DIR)
TRAIN_ANNOTATION_PATH = os.path.join(TFRECORD_DIR, "train_annotation.csv")

# Preprocessing image size
IMAGE_SIZE = (320, 320)


# Training paths and configurations
MODEL_DIR = os.path.join(BASE_DIR, "models", "helmet_detection_ssd_mnet_v2")
LOG_DIR = os.path.join(MODEL_DIR, 'logs')
PIPELINE_CONFIG_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "models", "helmet_detection_ssd_mnet_v2", "pipeline.config"))
# Training configurations
NUM_TRAIN_STEPS = 40000
SAMPLE_1_OF_N_EVAL_EXAMPLES = 1
CHECKPOINT_EVERY_N = 500
EVAL_INTERVAL = 500
SUMMARY_INTERVAL = 100

# Model and label map paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "exportedModel", "saved_model")
LABEL_PATH = os.path.join(BASE_DIR, "data", "processed", "label_map", "label_map.pbtxt")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "test_output", "test_video", "output_video.mp4")

# Webcam Detection settings
VIDEO_CODEC = 'XVID'
FRAME_RATE = 20.0
MIN_SCORE_THRESH = 0.5
MAX_BOXES = 10
