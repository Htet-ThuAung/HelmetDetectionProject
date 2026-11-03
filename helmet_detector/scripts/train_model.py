import sys
from pathlib import Path
import tensorflow as tf

# Resolve project paths independently of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELMET_DETECTOR_DIR = PROJECT_ROOT / "helmet_detector"

# Make both package roots available.
# The local Object Detection API must take priority over installed packages.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HELMET_DETECTOR_DIR))

from helmet_detector.config.config import (
    PIPELINE_CONFIG_PATH,
    MODEL_DIR,
    LOG_DIR,
    NUM_TRAIN_STEPS,
    CHECKPOINT_EVERY_N,
)


from object_detection import model_lib_v2
from object_detection.utils import config_util


def configure_gpu():
    """Enable GPU memory growth when a compatible GPU is available."""
    gpus = tf.config.experimental.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not enable GPU memory growth: {exc}")



def main():
    tf.get_logger().setLevel("INFO")

    configure_gpu()

    # Validate the pipeline configuration before training.
    configs = config_util.get_configs_from_pipeline_file(
        PIPELINE_CONFIG_PATH
    )

    model_type = configs["model"].WhichOneof("model")

    configured_steps = configs["train_config"].num_steps

    if model_type != "ssd":
        raise ValueError(f"Expected SSD model, got: {model_type}")

    num_classes = configs["model"].ssd.num_classes

    if num_classes != 2:
        raise ValueError(f"Expected 2 classes, got: {num_classes}")

    if configured_steps != NUM_TRAIN_STEPS:
        raise ValueError(
            f"Training step mismatch: config={configured_steps}, "
            f"script={NUM_TRAIN_STEPS}"
        )


    print(f"Project root: {PROJECT_ROOT}")
    print(f"Object Detection API: {model_lib_v2.__file__}")
    print(f"Pipeline config: {PIPELINE_CONFIG_PATH}")
    print(f"Model type: {model_type}")
    print(f"Number of classes: {num_classes}")
    print(f"Training steps: {configured_steps}")

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    print("Starting training...")

    model_lib_v2.train_loop(
        pipeline_config_path=PIPELINE_CONFIG_PATH,
        model_dir=MODEL_DIR,
        train_steps=NUM_TRAIN_STEPS,
        use_tpu=False,
        checkpoint_every_n=CHECKPOINT_EVERY_N,
        record_summaries=True,
    )


if __name__ == "__main__":
    main()
