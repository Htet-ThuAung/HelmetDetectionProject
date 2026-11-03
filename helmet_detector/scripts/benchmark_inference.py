
import csv
import json
import platform
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "raw" / "val" / "images"

MODELS = {
    "original": ROOT / "models" / "exportedModel" / "saved_model",
    "checkpoint76": (
        ROOT / "models" / "exportedModel_checkpoint76" / "saved_model"
    ),
    "checkpoint78": (
        ROOT / "models" / "exportedModel_checkpoint78" / "saved_model"
    ),
}

OUTPUT_DIR = ROOT / "test_output" / "performance_benchmark_cpu_round_robin"

# Benchmark settings
RANDOM_SEED = 42
NUM_IMAGES = 50
WARMUP_RUNS = 20
MEASUREMENT_ROUNDS = 5

# Execution device: CPU for the first experiment.
DEVICE = "/CPU:0"

# TensorFlow model configuration
SCORE_THRESHOLD = 0.5


# --------------------------------------------------
# Environment verification
# --------------------------------------------------

def verify_environment():
    """Verify TensorFlow and the requested execution device."""

    print("\n" + "=" * 60)
    print("BENCHMARK ENVIRONMENT")
    print("=" * 60)

    print("Python:", platform.python_version())
    print("Platform:", platform.platform())
    print("TensorFlow:", tf.__version__)
    print("Requested device:", DEVICE)

    physical_gpus = tf.config.list_physical_devices("GPU")

    print("Available GPUs:", physical_gpus)

    if DEVICE == "/GPU:0" and not physical_gpus:
        raise RuntimeError(
            "GPU execution requested, but TensorFlow cannot detect a GPU."
        )

    if DEVICE not in {"/CPU:0", "/GPU:0"}:
        raise ValueError(f"Unsupported device: {DEVICE}")

    print("Environment verification completed.")


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model(model_name, model_path):
    """Load a TensorFlow SavedModel and its serving signature."""

    saved_model_file = model_path / "saved_model.pb"

    if not saved_model_file.exists():
        raise FileNotFoundError(
            f"SavedModel not found for {model_name}: {model_path}"
        )

    print(f"Loading model: {model_name}")

    model = tf.saved_model.load(str(model_path))

    if "serving_default" not in model.signatures:
        raise ValueError(
            f"Missing serving_default signature: {model_name}"
        )

    infer = model.signatures["serving_default"]

    print(f"Successfully loaded: {model_name}")

    return infer




# --------------------------------------------------
# Image preparation
# --------------------------------------------------

def load_image(image_path):
    """Load an image and convert it to RGB."""

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return image_rgb


def prepare_tensor(image_rgb):
    """Convert an RGB image into a TensorFlow input tensor."""

    with tf.device(DEVICE):
        input_tensor = tf.convert_to_tensor(
            image_rgb[None, ...],
            dtype=tf.uint8,
        )

    return input_tensor


# --------------------------------------------------
# Inference functions
# --------------------------------------------------

def run_model(infer, input_tensor):
    """Run the model and synchronize all output tensors."""

    with tf.device(DEVICE):
        outputs = infer(input_tensor)

        # Synchronize every output before stopping the timer.
        for output in outputs.values():
            output.numpy()

    return outputs


def run_end_to_end(infer, image_path):
    """Measure image loading, preprocessing and model inference."""

    image_rgb = load_image(image_path)
    input_tensor = prepare_tensor(image_rgb)

    outputs = run_model(infer, input_tensor)

    # Extract predictions to include postprocessing in the measurement.
    count = int(outputs["num_detections"][0].numpy())

    boxes = outputs["detection_boxes"][0, :count].numpy()
    scores = outputs["detection_scores"][0, :count].numpy()
    classes = outputs["detection_classes"][0, :count].numpy()

    return boxes, scores, classes


# --------------------------------------------------
# Validation image selection
# --------------------------------------------------

def select_validation_images():
    """Select a reproducible subset of validation images."""

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    all_images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if len(all_images) < NUM_IMAGES:
        raise ValueError(
            f"Expected at least {NUM_IMAGES} images, "
            f"but found {len(all_images)}."
        )

    rng = random.Random(RANDOM_SEED)
    selected_images = rng.sample(all_images, NUM_IMAGES)

    print(f"\nSelected {len(selected_images)} validation images.")

    return selected_images


# --------------------------------------------------
# Latency statistics
# --------------------------------------------------

def calculate_latency_statistics(latencies):
    """Calculate descriptive statistics for latency measurements."""

    values = np.asarray(latencies, dtype=np.float64)

    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


# --------------------------------------------------
# Model benchmarking
# --------------------------------------------------


def prepare_benchmark_images(image_paths):
    """Load and prepare images before model-only timing."""
    prepared_images = []

    for image_path in image_paths:
        image_rgb = load_image(image_path)
        input_tensor = prepare_tensor(image_rgb)

        prepared_images.append({
            "path": image_path,
            "tensor": input_tensor,
        })

    return prepared_images


def benchmark_one_round(infer, prepared_images):
    """Measure one round of model-only and end-to-end inference."""
    model_latencies = []
    end_to_end_latencies = []

    for item in prepared_images:
        # Model-only inference
        start = time.perf_counter()
        run_model(infer, item["tensor"])
        model_elapsed_ms = (time.perf_counter() - start) * 1000
        model_latencies.append(model_elapsed_ms)

        # End-to-end inference
        start = time.perf_counter()
        run_end_to_end(infer, item["path"])
        end_to_end_elapsed_ms = (time.perf_counter() - start) * 1000
        end_to_end_latencies.append(end_to_end_elapsed_ms)

    return model_latencies, end_to_end_latencies


# --------------------------------------------------
# Environment metadata
# --------------------------------------------------

def collect_environment():
    """Collect hardware and software information."""

    import subprocess

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow": tf.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "requested_device": DEVICE,
        "available_gpus": [
            device.name
            for device in tf.config.list_physical_devices("GPU")
        ],
        "available_cpus": [
            device.name
            for device in tf.config.list_physical_devices("CPU")
        ],
    }

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name"
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        environment["cpu_model"] = result.stdout.strip()

    except (subprocess.SubprocessError, OSError):
        environment["cpu_model"] = "Not available"

    return environment


# --------------------------------------------------
# Save benchmark results
# --------------------------------------------------

def save_results(results, environment):
    """Save benchmark results in JSON and CSV formats."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Environment information.
    environment_path = OUTPUT_DIR / "benchmark_environment.json"

    with environment_path.open("w", encoding="utf-8") as file:
        json.dump(environment, file, indent=4)

    # Complete benchmark results.
    summary_path = OUTPUT_DIR / "benchmark_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Flattened CSV.
    csv_path = OUTPUT_DIR / "benchmark_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "model",
            "device",
            "images",
            "warmup_runs",
            "measurement_rounds",
            "model_mean_ms",
            "model_median_ms",
            "model_p95_ms",
            "model_std_ms",
            "model_throughput_fps",
            "end_to_end_mean_ms",
            "end_to_end_median_ms",
            "end_to_end_p95_ms",
            "end_to_end_std_ms",
            "end_to_end_throughput_fps",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "model": result["model"],
                "device": result["device"],
                "images": result["images"],
                "warmup_runs": result["warmup_runs"],
                "measurement_rounds": result["measurement_rounds"],

                "model_mean_ms": result["model_only"]["mean_ms"],
                "model_median_ms": result["model_only"]["median_ms"],
                "model_p95_ms": result["model_only"]["p95_ms"],
                "model_std_ms": result["model_only"]["std_ms"],
                "model_throughput_fps": result["model_throughput_fps"],

                "end_to_end_mean_ms": result["end_to_end"]["mean_ms"],
                "end_to_end_median_ms": result["end_to_end"]["median_ms"],
                "end_to_end_p95_ms": result["end_to_end"]["p95_ms"],
                "end_to_end_std_ms": result["end_to_end"]["std_ms"],
                "end_to_end_throughput_fps": result["end_to_end_throughput_fps"],
            })

    print("\nBenchmark reports saved to:", OUTPUT_DIR)


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    verify_environment()
    environment = collect_environment()
    image_paths = select_validation_images()

    # Load all models before starting measurements.
    infer_by_model = {}
    prepared_by_model = {}

    for model_name, model_path in MODELS.items():
        infer = load_model(model_name, model_path)
        infer_by_model[model_name] = infer

        prepared_by_model[model_name] = (
            prepare_benchmark_images(image_paths)
        )

    # Warm up every model before collecting measurements.
    print("\nWarming up all models...")

    for model_name, infer in infer_by_model.items():
        warmup_tensor = prepared_by_model[model_name][0]["tensor"]

        for _ in range(WARMUP_RUNS):
            run_model(infer, warmup_tensor)

        print(f"Warm-up completed: {model_name}")

    # Store measurements separately for each model.
    measurements = {
        model_name: {
            "model_only": [],
            "end_to_end": [],
        }
        for model_name in MODELS
    }

    # Rotate model order in every measurement round.
    rng = random.Random(RANDOM_SEED)

    model_names = list(MODELS.keys())

    for round_number in range(MEASUREMENT_ROUNDS):
        round_order = model_names.copy()
        rng.shuffle(round_order)

        print(
            f"\nMeasurement round "
            f"{round_number + 1}/{MEASUREMENT_ROUNDS}"
        )
        print("Model order:", " -> ".join(round_order))

        for model_name in round_order:
            model_latencies, end_to_end_latencies = (
                benchmark_one_round(
                    infer_by_model[model_name],
                    prepared_by_model[model_name],
                )
            )

            measurements[model_name]["model_only"].extend(
                model_latencies
            )
            measurements[model_name]["end_to_end"].extend(
                end_to_end_latencies
            )

    # Calculate final statistics.
    results = []

    for model_name in model_names:
        model_stats = calculate_latency_statistics(
            measurements[model_name]["model_only"]
        )

        end_to_end_stats = calculate_latency_statistics(
            measurements[model_name]["end_to_end"]
        )

        results.append({
            "model": model_name,
            "device": DEVICE,
            "images": len(image_paths),
            "warmup_runs": WARMUP_RUNS,
            "measurement_rounds": MEASUREMENT_ROUNDS,
            "model_only": model_stats,
            "end_to_end": end_to_end_stats,
            "model_throughput_fps": (
                1000 / model_stats["mean_ms"]
                if model_stats["mean_ms"] > 0 else 0.0
            ),
            "end_to_end_throughput_fps": (
                1000 / end_to_end_stats["mean_ms"]
                if end_to_end_stats["mean_ms"] > 0 else 0.0
            ),
        })

    save_results(results, environment)

    print("\n" + "=" * 60)
    print("ROUND-ROBIN CPU BENCHMARK COMPLETED")
    print("=" * 60)

    for result in results:
        print(f"\nModel: {result['model']}")
        print(
            "Model-only mean:",
            f"{result['model_only']['mean_ms']:.2f} ms"
        )
        print(
            "Model-only median:",
            f"{result['model_only']['median_ms']:.2f} ms"
        )
        print(
            "Model-only P95:",
            f"{result['model_only']['p95_ms']:.2f} ms"
        )
        print(
            "End-to-end mean:",
            f"{result['end_to_end']['mean_ms']:.2f} ms"
        )
        print(
            "End-to-end median:",
            f"{result['end_to_end']['median_ms']:.2f} ms"
        )
        print(
            "End-to-end P95:",
            f"{result['end_to_end']['p95_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()

