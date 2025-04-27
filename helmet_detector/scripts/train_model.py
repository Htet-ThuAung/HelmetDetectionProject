import os
import tensorflow as tf
from helmet_detector.object_detection import model_lib_v2  # Import from your package
from helmet_detector.object_detection.utils import config_util  # Import from your package
from tensorflow.keras.callbacks import TensorBoard
from helmet_detector.config.config import *  # Import from the updated config.py

# Enable GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
import os

# CONFIGURATION
PIPELINE_CONFIG_PATH = PIPELINE_CONFIG_PATH  # Loaded from config.py
MODEL_DIR = MODEL_DIR  # Loaded from config.py
LOG_DIR = LOG_DIR  # Loaded from config.py

NUM_TRAIN_STEPS = NUM_TRAIN_STEPS  # Loaded from config.py
SAMPLE_1_OF_N_EVAL_EXAMPLES = SAMPLE_1_OF_N_EVAL_EXAMPLES  # Loaded from config.py
print(f"Resolved PIPELINE_CONFIG_PATH: {PIPELINE_CONFIG_PATH}")
import os
print(f"Current working directory: {os.getcwd()}")

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Load model configs
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG_PATH)

# TensorBoard Logging
log_dir = os.path.join(LOG_DIR, 'tensorboard_logs')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Logging config
tf.get_logger().setLevel('INFO')  # Default INFO level shows step logs

# Start Training...
def main():
    print(f"Loaded pipeline config from: {PIPELINE_CONFIG_PATH}")
    print(f"Training model for {NUM_TRAIN_STEPS} steps...")

    model_lib_v2.train_loop(
        pipeline_config_path=PIPELINE_CONFIG_PATH,
        model_dir=MODEL_DIR,
        config_override=None,
        train_steps=NUM_TRAIN_STEPS,
        use_tpu=False,
        checkpoint_every_n=CHECKPOINT_EVERY_N,  # Using the value from config.py
        record_summaries=True,  # Ensure summaries are recorded for TensorBoard
        tensorboard_callback=tensorboard_callback,  # Pass TensorBoard callback
        eval_interval=EVAL_INTERVAL,  # Using the value from config.py
        summary_interval=SUMMARY_INTERVAL  # Using the value from config.py
    )

if __name__ == '__main__':
    main()
