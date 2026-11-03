from pathlib import Path
import csv

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.framework import tensor_util


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = (
    PROJECT_ROOT
    / "helmet_detector"
    / "models"
    / "helmet_detection_ssd_mnet_v2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "helmet_detector"
    / "test_output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "tensorboard_metrics.csv"

event_files = sorted(LOG_DIR.rglob("events.out.tfevents.*"))

rows = []

for event_file in event_files:
    relative_path = event_file.relative_to(PROJECT_ROOT)

    accumulator = EventAccumulator(
        str(event_file),
        size_guidance={
            "scalars": 0,
            "tensors": 0,
        },
    )
    accumulator.Reload()

    print(f"\n{'=' * 70}")
    print(f"FILE: {relative_path}")

    tags = accumulator.Tags()

    # Extract traditional scalar summaries.
    for tag in tags.get("scalars", []):
        events = accumulator.Scalars(tag)

        for event in events:
            rows.append({
                "event_file": str(relative_path),
                "log_directory": event_file.parent.name,
                "summary_type": "scalar",
                "metric": tag,
                "step": event.step,
                "value": event.value,
            })

        print(f"{tag}: {len(events)} scalar records")

    # Extract scalar values stored as tensor summaries.
    for tag in tags.get("tensors", []):
        if tag == "train_input_images" or tag.startswith("eval_side_by_side"):
            continue

        events = accumulator.Tensors(tag)
        extracted_count = 0

        for event in events:
            array = tensor_util.MakeNdarray(event.tensor_proto)

            # Only retain tensors containing exactly one value.
            if array.size != 1:
                continue

            value = float(array.reshape(-1)[0])

            rows.append({
                "event_file": str(relative_path),
                "log_directory": event_file.parent.name,
                "summary_type": "tensor",
                "metric": tag,
                "step": event.step,
                "value": value,
            })

            extracted_count += 1

        if extracted_count:
            print(f"{tag}: {extracted_count} tensor records")

if rows:
    fieldnames = [
        "event_file",
        "log_directory",
        "summary_type",
        "metric",
        "step",
        "value",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 70}")
    print(f"Total extracted records: {len(rows)}")
    print(f"Saved detailed metrics: {CSV_PATH}")

else:
    print("\nNo scalar metric values were extracted.")