
import tensorflow as tf
from pathlib import Path

project_root = Path(__file__).resolve().parent
models_dir = project_root / "helmet_detector" / "models"

model_names = [
    "exportedModel",
    "exportedModel_checkpoint76",
]

for name in model_names:
    model_path = models_dir / name / "saved_model"

    print(f"\n{'=' * 50}")
    print(f"Model: {name}")

    model = tf.saved_model.load(str(model_path))

    print("Signatures:", list(model.signatures.keys()))
    print("Callable:", callable(model))
