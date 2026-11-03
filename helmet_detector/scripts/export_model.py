
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "helmet_detection_ssd_mnet_v2"
MODEL_DIR = PROJECT_ROOT / "helmet_detector" / "models" / MODEL_NAME

PIPELINE_CONFIG_PATH = MODEL_DIR / "pipeline.config"

CHECKPOINT_DIR = MODEL_DIR / "best_checkpoint_78"
CHECKPOINT_NAME = "ckpt-78"

EXPORT_DIR = (
    PROJECT_ROOT
    / "helmet_detector"
    / "models"
    / "exportedModel_checkpoint78"
)


def main():
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_NAME

    print("\n[INFO] Preparing model export...")
    print(f"Pipeline config : {PIPELINE_CONFIG_PATH}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Export directory: {EXPORT_DIR}")

    required_files = [
        PIPELINE_CONFIG_PATH,
        CHECKPOINT_DIR / "checkpoint",
        Path(str(checkpoint_path) + ".index"),
        Path(str(checkpoint_path) + ".data-00000-of-00001"),
    ]

    missing_files = [
        str(path) for path in required_files if not path.is_file()
    ]

    if missing_files:
        print("\n[ERROR] Required files are missing:")
        for path in missing_files:
            print(f" - {path}")
        return 1

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "helmet_detector.object_detection.exporter_main_v2",
        "--input_type=image_tensor",
        f"--pipeline_config_path={PIPELINE_CONFIG_PATH}",
        f"--trained_checkpoint_dir={CHECKPOINT_DIR}",
        f"--output_directory={EXPORT_DIR}",
    ]

    print("\n[INFO] Running TensorFlow Object Detection exporter...\n")

    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
        )

    except subprocess.CalledProcessError as error:
        print("\n[ERROR] Model export failed.")
        print(f"Exporter exited with code: {error.returncode}")
        return 1

    saved_model_path = EXPORT_DIR / "saved_model" / "saved_model.pb"

    if not saved_model_path.is_file():
        print("\n[ERROR] Export finished, but SavedModel was not found.")
        return 1


    print("\n[SUCCESS] Model exported successfully!")
    print(f"SavedModel: {EXPORT_DIR / 'saved_model'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())