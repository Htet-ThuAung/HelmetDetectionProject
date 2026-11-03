
import os
import csv
import tensorflow as tf

from helmet_detector.object_detection import model_lib_v2
from helmet_detector.object_detection import inputs
from helmet_detector.config.config import (
    PIPELINE_CONFIG_PATH,
    MODEL_DIR,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_STEPS = range(75, 82)

CHECKPOINT_DIR = MODEL_DIR

EVALUATION_DIR = os.path.join(
    MODEL_DIR, "checkpoint_evaluations"
)

CSV_PATH = os.path.join(
    EVALUATION_DIR, "checkpoint_comparison.csv"
)

os.makedirs(EVALUATION_DIR, exist_ok=True)


# --------------------------------------------------
# Load pipeline configuration
# --------------------------------------------------

configs = (
    model_lib_v2.MODEL_BUILD_UTIL_MAP[
        "get_configs_from_pipeline_file"
    ](PIPELINE_CONFIG_PATH)
)

model_config = configs["model"]
train_config = configs["train_config"]
eval_config = configs["eval_config"]
eval_input_config = configs["eval_input_configs"][0]

eval_input_config.num_epochs = 1
eval_input_config.shuffle = False


# --------------------------------------------------
# Build evaluation model and dataset
# --------------------------------------------------

strategy = tf.compat.v2.distribute.get_strategy()

with strategy.scope():
    detection_model = (
        model_lib_v2.MODEL_BUILD_UTIL_MAP[
            "detection_model_fn_base"
        ](
            model_config=model_config,
            is_training=True,
        )
    )

eval_dataset = strategy.experimental_distribute_dataset(
    inputs.eval_input(
        eval_config=eval_config,
        eval_input_config=eval_input_config,
        model_config=model_config,
        model=detection_model,
    )
)

# Build optimizer for checkpoint restoration.
global_step = tf.compat.v2.Variable(
    0, trainable=False, dtype=tf.int64
)

optimizer, _ = model_lib_v2.optimizer_builder.build(
    train_config.optimizer,
    global_step=global_step,
)


# --------------------------------------------------
# Evaluate individual checkpoints
# --------------------------------------------------

results = []

for checkpoint_number in CHECKPOINT_STEPS:

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"ckpt-{checkpoint_number}",
    )

    print("\n" + "=" * 60)
    print(f"Evaluating checkpoint {checkpoint_number}")
    print("=" * 60)

    if not tf.io.gfile.exists(checkpoint_path + ".index"):
        print("Checkpoint not found. Skipping.")
        continue

    # Restore model and optimizer state.
    checkpoint = tf.train.Checkpoint(
        step=global_step,
        model=detection_model,
        optimizer=optimizer,
    )

    checkpoint.restore(checkpoint_path).expect_partial()

    print("Training step:", global_step.numpy())

    # Run evaluation.
    metrics = model_lib_v2.eager_eval_loop(
        detection_model=detection_model,
        configs=configs,
        eval_dataset=eval_dataset,
        use_tpu=False,
        global_step=global_step,
    )

    row = {
        "checkpoint": checkpoint_number,
        "training_step": int(global_step.numpy()),
    }

    for name, value in metrics.items():
        row[name] = float(value.numpy()) if hasattr(value, "numpy") else float(value)

    results.append(row)

    print("Evaluation complete.")

    for name, value in row.items():
        print(f"{name}: {value}")


# --------------------------------------------------
# Save results
# --------------------------------------------------

if results:
    fieldnames = list(
        dict.fromkeys(
            key for row in results for key in row
        )
    )

    with open(
        CSV_PATH, "w", newline="", encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", CSV_PATH)

    map_key = "DetectionBoxes_Precision/mAP"

    if all(map_key in row for row in results):
        best = max(results, key=lambda row: row[map_key])

        print("\nBest checkpoint:")
        print("Checkpoint:", best["checkpoint"])
        print("Training step:", best["training_step"])
        print("mAP:", best[map_key])
else:
    print("No checkpoints were evaluated.")
