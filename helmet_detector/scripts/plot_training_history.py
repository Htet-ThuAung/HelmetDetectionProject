
from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "helmet_detector"
    / "test_output"
    / "tensorboard_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "helmet_detector"
    / "results"
    / "training_plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load TensorBoard metrics
# --------------------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"TensorBoard metrics CSV not found: {CSV_PATH}\n"
        "Run inspect_tensorboard_logs.py first."
    )

training_data = defaultdict(lambda: defaultdict(list))
evaluation_data = defaultdict(lambda: defaultdict(list))

with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    required_columns = {
        "event_file",
        "log_directory",
        "metric",
        "step",
        "value",
    }

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError(
            "The TensorBoard CSV is missing required columns."
        )

    for row in reader:
        event_file = row["event_file"]
        metric = row["metric"]
        step = int(row["step"])
        value = float(row["value"])

        if row["log_directory"] == "train":
            training_data[metric][event_file].append(
                (step, value)
            )

        elif row["log_directory"] == "eval":
            evaluation_data[metric][event_file].append(
                (step, value)
            )


training_event_files = {
    event_file
    for metric_data in training_data.values()
    for event_file in metric_data
}

evaluation_event_files = {
    event_file
    for metric_data in evaluation_data.values()
    for event_file in metric_data
}

print(f"Training event files: {len(training_event_files)}")
print(f"Evaluation event files: {len(evaluation_event_files)}")


# --------------------------------------------------
# Plot training metrics by event file
# --------------------------------------------------

def plot_training_metric(metric, filename, title, ylabel):
    log_segments = training_data.get(metric, {})

    if not log_segments:
        print(f"WARNING: No data found for {metric}")
        return

    plt.figure(figsize=(12, 6))

    ordered_segments = sorted(
        log_segments.items(),
        key=lambda item: (
            min(step for step, _ in item[1]),
            item[0],
        ),
    )

    for index, (event_file, values) in enumerate(
        ordered_segments,
        start=1,
    ):
        sorted_values = sorted(values, key=lambda item: item[0])

        steps = [step for step, _ in sorted_values]
        scores = [value for _, value in sorted_values]

        event_name = Path(event_file).name

        plt.plot(
            steps,
            scores,
            marker=".",
            markersize=3,
            linewidth=1.2,
            label=f"Log segment {index}: {event_name}",
        )

    plt.title(title)
    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved: {output_path}")


plot_training_metric(
    "Loss/total_loss",
    "training_total_loss_by_log_segment.png",
    "Training Total Loss by Log Segment",
    "Total loss",
)

plot_training_metric(
    "Loss/classification_loss",
    "training_classification_loss_by_log_segment.png",
    "Classification Loss by Log Segment",
    "Loss",
)

plot_training_metric(
    "Loss/localization_loss",
    "training_localization_loss_by_log_segment.png",
    "Localization Loss by Log Segment",
    "Loss",
)

plot_training_metric(
    "learning_rate",
    "learning_rate_schedule.png",
    "Learning Rate Schedule",
    "Learning rate",
)

plot_training_metric(
    "steps_per_sec",
    "training_throughput.png",
    "Training Throughput",
    "Steps per second",
)


# --------------------------------------------------
# Export evaluation metrics
# --------------------------------------------------

evaluation_tags = [
    "DetectionBoxes_Precision/mAP",
    "DetectionBoxes_Precision/mAP@.50IOU",
    "DetectionBoxes_Precision/mAP@.75IOU",
]

evaluation_csv = OUTPUT_DIR / "evaluation_map_values.csv"

with evaluation_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "event_file",
        "step",
        "metric",
        "value",
    ])

    for metric in evaluation_tags:
        for event_file, values in sorted(
            evaluation_data.get(metric, {}).items()
        ):
            for step, value in sorted(values):
                writer.writerow([
                    event_file,
                    step,
                    metric,
                    value,
                ])

print(f"Saved: {evaluation_csv}")


# --------------------------------------------------
# Plot evaluation metrics
# --------------------------------------------------

metric_labels = {
    "DetectionBoxes_Precision/mAP":
        "mAP (IoU 0.50:0.95)",

    "DetectionBoxes_Precision/mAP@.50IOU":
        "mAP@0.50 IoU",

    "DetectionBoxes_Precision/mAP@.75IOU":
        "mAP@0.75 IoU",
}

plt.figure(figsize=(11, 6))

has_evaluation_data = False

for metric in evaluation_tags:
    event_files = evaluation_data.get(metric, {})

    if not event_files:
        continue

    has_evaluation_data = True

    # Keep unique coordinates for visualization only.
    # The original event-file records remain in the CSV.
    unique_points = {
        (step, value)
        for values in event_files.values()
        for step, value in values
    }

    points = sorted(unique_points)

    steps = [step for step, _ in points]
    scores = [value * 100 for _, value in points]

    plt.scatter(
        steps,
        scores,
        s=45,
        label=metric_labels[metric],
    )

if has_evaluation_data:
    evaluation_steps = sorted({
        step
        for metric in evaluation_tags
        for values in evaluation_data.get(metric, {}).values()
        for step, _ in values
    })

    plt.title("Evaluation Metrics During Training")
    plt.xlabel("Training steps")
    plt.ylabel("Average precision (%)")
    plt.xticks(evaluation_steps)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "evaluation_map_comparison.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Saved: {output_path}")

else:
    plt.close()
    print("WARNING: No evaluation metrics found.")


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nPlot generation completed.")
print(f"Output directory: {OUTPUT_DIR}")

for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"  {path.name}")
