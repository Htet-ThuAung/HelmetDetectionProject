import os
import sys
import subprocess

# Add the root folder of my_package to the system path
sys.path.append(r'D:\Projects\HelmetDetectionProject\helmet_detector')

# === Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_NAME = 'helmet_detection_ssd_mnet_v2'
MODEL_DIR = os.path.join(BASE_DIR, 'models', MODEL_NAME)

PIPELINE_CONFIG_PATH = os.path.join(MODEL_DIR, 'pipeline.config')
CHECKPOINT_DIR = MODEL_DIR
CHECKPOINT_NAME = 'ckpt-81'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
EXPORT_DIR = os.path.join(BASE_DIR, 'models', 'exportedModel')

def main():
    print(f"\n[INFO] Exporting model...")
    print(f"Pipeline config: {PIPELINE_CONFIG_PATH}")
    print(f"Checkpoint path : {CHECKPOINT_PATH}")
    print(f"Export to       : {EXPORT_DIR}\n")

    if not os.path.exists(CHECKPOINT_PATH + '.index'):
        print(f"Checkpoint file '{CHECKPOINT_PATH}' not found. Cannot export model.")
        return

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Use subprocess to call the exporter
    command = [
        sys.executable,
        '-m', 'helmet_detector.object_detection.exporter_main_v2',
        '--input_type=image_tensor',
        f'--pipeline_config_path={PIPELINE_CONFIG_PATH}',
        f'--trained_checkpoint_dir={CHECKPOINT_DIR}',  # Correct flag
        f'--output_directory={EXPORT_DIR}',
    ]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with error:\n{e.stderr}")

    print("Model export completed!")

if __name__ == '__main__':
    main()
