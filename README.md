# Helmet Detection using TensorFlow Object Detection API

A computer vision project for detecting **helmet** and **no-helmet** instances using a custom-trained SSD MobileNet V2 FPNLite object detection model.

Built with TensorFlow 2 and the TensorFlow Object Detection API, this project covers the complete object detection workflow: dataset preparation, Pascal VOC annotation processing, TFRecord generation, model fine-tuning, checkpoint evaluation, SavedModel export, and image/webcam inference.

## Project Overview

The model detects two classes:

- `helmet`
- `no_helmet`

It uses a custom Pascal VOC-annotated dataset and a COCO-pretrained SSD MobileNet V2 FPNLite model.

The exported model supports single-image inference and webcam-based detection with bounding boxes, class labels, and confidence scores.

## Features

- Custom object detection using SSD MobileNet V2 FPNLite
- Pascal VOC XML annotation processing
- TFRecord generation for training and validation
- TensorFlow Object Detection API training pipeline
- Checkpoint evaluation and model comparison
- SavedModel export
- Single-image inference
- Webcam inference
- Annotated image and video output
- Training and evaluation analysis scripts
- CPU and GPU inference performance benchmarking

## Model Architecture

| Component | Configuration |
|---|---|
| Architecture | SSD MobileNet V2 FPNLite |
| Input resolution | 320 × 320 |
| Framework | TensorFlow 2.10.0 |
| Detection framework | TensorFlow Object Detection API |
| Pretrained initialization | COCO 2017 checkpoint |
| Number of classes | 2 |
| Training steps | 40,000 |
| Batch size | 8 |

The model was fine-tuned from a pretrained TensorFlow Model Zoo checkpoint using a custom detection pipeline configuration.

## Dataset

The dataset contains manually annotated images in Pascal VOC XML format.

### Dataset size

| Split | Original images | TFRecord examples |
|---|---:|---:|
| Training | 572 | 570 |
| Validation | 236 | 236 |

Two training annotation files contain no object annotations and are excluded during TFRecord generation.

Annotations were created using [LabelImg](https://github.com/HumanSignal/labelImg).

### Class distribution

| Class | Training objects | Validation objects |
|---|---:|---:|
| Helmet | 314 | 112 |
| No helmet | 256 | 124 |

These figures represent annotated objects, not the number of images containing each class.

### Data preparation

1. Read images and Pascal VOC XML annotations.
2. Validate annotation coordinates and class labels.
3. Convert annotations into the required training representation.
4. Generate TensorFlow TFRecord files.
5. Verify generated records against the original annotations.

The dataset is not included in the Git repository.

## Training Configuration

The model was fine-tuned using the TensorFlow Object Detection API.

Important training settings include:

- Batch size: 8
- Training steps: 40,000
- Input resolution: 320 × 320
- Horizontal flip and random crop augmentation
- Cosine decay learning-rate schedule
- Momentum optimizer

Training and evaluation summaries were recorded using TensorBoard.

Some TensorBoard event files contain overlapping training steps with different logged values. These log segments are treated separately in the training visualizations rather than being merged into a misleading continuous curve.

## Checkpoint Selection

Several late-stage checkpoints were evaluated.

Among checkpoints 75–81, checkpoint 76 achieved the highest native TensorFlow Object Detection API mAP:

| Checkpoint | Native evaluation mAP |
|---|---:|
| 75 | 0.6908 |
| 76 | 0.6953 |
| 77 | 0.6772 |
| 78 | 0.6876 |
| 79 | 0.6841 |
| 80 | 0.6843 |
| 81 | 0.6872 |

Checkpoint 76 was selected for additional comparison.

This selection refers to the evaluated checkpoints 75–81, not necessarily every checkpoint produced during training.

## Model Evaluation

A custom evaluation script evaluates the exported detection models against the 236 validation images.

### Evaluation methodology

- Confidence threshold: 0.50
- Matching IoU threshold for precision and recall: 0.50
- AP IoU thresholds: 0.50–0.95
- AP calculation: 101-point interpolated precision
- Precision, recall, and F1-score are calculated at the fixed confidence and IoU thresholds.

The custom evaluation metrics below are separate from the native TensorFlow Object Detection API evaluation metrics above. They use different evaluation implementations and should not be compared directly.

### Custom evaluation results

| Metric | Original model | Checkpoint 76 |
|---|---:|---:|
| Precision | 96.76% | 96.71% |
| Recall | 88.56% | 87.29% |
| F1-score | 92.48% | 91.76% |
| mAP@0.50 | 96.22% | 96.26% |
| mAP@0.50:0.95 | 75.18% | 75.72% |
| Mean processing time | 120.92 ms | 108.67 ms |
| Median processing time | 103.87 ms | 98.61 ms |

The original model and checkpoint 76 show similar overall detection performance.

Checkpoint 76 has a slightly higher mAP@0.50:0.95, while the original model has higher recall and F1-score at the selected operating threshold.

### Evaluation observations

- Helmet detection has higher recall than no-helmet detection.
- No-helmet localization performance decreases more at stricter IoU thresholds.
- The results indicate opportunities to improve localization consistency and detection recall.

### Timing notes

Processing time includes image disk reading, preprocessing, tensor creation, model inference, and output conversion.

The measurements were collected without a dedicated warm-up or controlled benchmarking procedure. They should therefore be treated as indicative end-to-end processing times, not hardware-independent model inference benchmarks.

All reported custom evaluation metrics use the validation dataset. The project does not currently have a separate held-out test set.

## Performance Benchmark

### Overview

Dedicated CPU and GPU inference benchmarks were implemented to compare three exported SSD MobileNet V2 FPNLite models:

- Original model
- Checkpoint 76
- Checkpoint 78

The benchmarks measure model-only inference latency and end-to-end inference latency using images from the validation dataset.

The objective is to understand how different model versions perform under the same hardware and software environment.

### Benchmark methodology

Both CPU and GPU benchmarks use the following configuration:

| Parameter | Value |
|---|---:|
| Validation images | 50 |
| Random seed | 42 |
| Batch size | 1 |
| Warm-up iterations | 20 per model |
| Measurement rounds | 5 |
| Measured inferences | 250 per model per timing category |

Two timing categories are evaluated.

**Model-only latency**

Measures the SavedModel inference call using image tensors prepared before timing. Output tensors are synchronized before the timer stops.

**End-to-end latency**

Includes:

1. Image loading.
2. BGR-to-RGB color conversion.
3. Tensor conversion.
4. Model inference.
5. Detection output extraction.

Performance statistics include mean, median, 95th-percentile latency (P95), standard deviation, and reciprocal-latency FPS estimates.

### CPU benchmark environment

| Component | Configuration |
|---|---|
| CPU | AMD Ryzen 5 3550H with Radeon Vega Mobile Gfx |
| Operating System | Windows 10 |
| Python | 3.10.10 |
| TensorFlow | 2.10.0 |
| OpenCV | 4.11.0 |
| NumPy | 1.23.5 |
| Execution device | CPU (`/CPU:0`) |
| TensorFlow GPU availability | No GPU detected in this environment |

### CPU benchmark results

Two benchmark experiments were conducted.

**Experiment 1 — Sequential model execution**

The models were evaluated sequentially, with each model completing its measurement rounds before the next model started.

| Model | Model-only mean | Model-only P95 | End-to-end mean | End-to-end P95 |
|---|---:|---:|---:|---:|
| Original | 102.42 ms | 110.59 ms | 110.13 ms | 118.54 ms |
| Checkpoint 76 | 102.73 ms | 112.84 ms | 109.41 ms | 119.45 ms |
| Checkpoint 78 | 111.86 ms | 123.73 ms | 119.56 ms | 132.90 ms |

**Experiment 2 — Shuffled model execution order**

The second experiment shuffled model execution order between measurement rounds to reduce potential ordering bias.

| Model | Model-only mean | Model-only P95 | End-to-end mean | End-to-end P95 |
|---|---:|---:|---:|---:|
| Original | 109.53 ms | 121.93 ms | **117.69 ms** | 132.12 ms |
| Checkpoint 76 | 111.44 ms | 128.43 ms | 119.73 ms | 139.92 ms |
| Checkpoint 78 | 110.99 ms | 122.79 ms | 119.52 ms | 136.36 ms |

Lower latency indicates faster inference.

### Estimated CPU end-to-end inference speed

For the second experiment, reciprocal-latency FPS estimates were calculated using:

`FPS = 1000 / mean latency (ms)`

| Model | Estimated FPS |
|---|---:|
| Original | **8.50** |
| Checkpoint 76 | 8.35 |
| Checkpoint 78 | 8.37 |

These values are reciprocal-latency estimates, not independently measured sustained throughput.

### CPU timing variability

Standard deviation of individual inference measurements from the second experiment:

| Model | Model-only std. dev. | End-to-end std. dev. |
|---|---:|---:|
| Original | 6.48 ms | 7.40 ms |
| Checkpoint 76 | 11.06 ms | 12.93 ms |
| Checkpoint 78 | 7.78 ms | 9.70 ms |

The original model demonstrated the lowest timing variability in the second experiment.

### GPU benchmark environment

GPU benchmarking was performed in a separate Conda environment to avoid changing the original CPU benchmark environment.

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce GTX 1050 |
| Dedicated GPU memory | 3 GB |
| CPU | AMD Ryzen 5 3550H with Radeon Vega Mobile Gfx |
| Operating System | Windows 10 |
| Python | 3.10.22 |
| TensorFlow | 2.10.0 |
| OpenCV | 4.10.0 |
| NumPy | 1.23.5 |
| NVIDIA driver | 512.78 |
| CUDA Toolkit | 11.2.2 |
| cuDNN | 8.1.0 |
| Execution device | GPU (`/GPU:0`) |

TensorFlow successfully detected the GTX 1050 and executed GPU operations. GPU placement was also verified during inference.

### GPU benchmark results

Two three-model benchmark sessions were conducted using the same benchmark configuration.

All three SavedModels were loaded into the same process. Model execution order was shuffled between measurement rounds, although the ordering was not guaranteed to be perfectly balanced.

**Experiment 1 — GPU benchmark session 1**

| Model | Model-only mean | Model-only P95 | End-to-end mean | End-to-end P95 |
|---|---:|---:|---:|---:|
| Original | 67.10 ms | 87.63 ms | 80.23 ms | 99.33 ms |
| Checkpoint 76 | 64.76 ms | 80.94 ms | 77.39 ms | 97.99 ms |
| Checkpoint 78 | 65.25 ms | 81.36 ms | 78.03 ms | 96.45 ms |

**Experiment 2 — GPU benchmark session 2**

| Model | Model-only mean | Model-only P95 | End-to-end mean | End-to-end P95 |
|---|---:|---:|---:|---:|
| Original | 78.98 ms | 98.10 ms | 94.08 ms | 115.25 ms |
| Checkpoint 76 | 73.78 ms | 93.67 ms | 88.79 ms | 111.36 ms |
| Checkpoint 78 | 77.76 ms | 97.38 ms | 92.00 ms | 111.86 ms |

### GPU results across both sessions

The following table reports the arithmetic average of the two session-level mean latencies. These averages summarize the observed sessions; they are not a substitute for additional independent benchmark repetitions.

| Model | Average model-only mean | Average end-to-end mean |
|---|---:|---:|
| Original | 73.04 ms | 87.16 ms |
| Checkpoint 76 | **69.27 ms** | **83.09 ms** |
| Checkpoint 78 | 71.51 ms | 85.01 ms |

Checkpoint 76 had the lowest average latency in both GPU timing categories across these two sessions.

However, the session-level results varied noticeably. In particular, the second session was slower than the first for all three models. This variation means that the observed differences should not be treated as statistically conclusive.

### Preliminary CPU and GPU comparison

The following table compares the shuffled-order CPU experiment with the first three-model GPU experiment.

| Model | CPU model-only | GPU model-only | CPU end-to-end | GPU end-to-end |
|---|---:|---:|---:|---:|
| Original | 109.53 ms | 67.10 ms | 117.69 ms | 80.23 ms |
| Checkpoint 76 | 111.44 ms | 64.76 ms | 119.73 ms | 77.39 ms |
| Checkpoint 78 | 110.99 ms | 65.25 ms | 119.52 ms | 78.03 ms |

The first GPU session showed lower latency than the shuffled-order CPU experiment for all three models.

These are preliminary cross-session comparisons, not a controlled CPU-versus-GPU experiment. The results may be affected by different runtime conditions, GPU temperature, clock behavior, background activity, and separate benchmark sessions.

### GPU monitoring observations

GPU telemetry was recorded at approximately one-second intervals during the second GPU benchmark session.

Observed values included:

| Measurement | Observation |
|---|---:|
| Idle GPU temperature before model loading | Approximately 62–63°C |
| Maximum sampled GPU temperature | 85°C |
| Peak reported GPU memory usage | 2,135 MiB |
| Reported dedicated GPU memory | 3,072 MiB |
| Peak sampled GPU utilization | 66% |

GPU temperature increased during the benchmark, while reported graphics clocks and performance states varied during the workload.

These observations show that GPU operating conditions changed during the benchmark. However, the telemetry alone does not establish that thermal throttling occurred. A dedicated throttle-reason investigation and controlled repeat experiments would be needed to confirm that.

### Performance analysis

The CPU experiments showed relatively similar end-to-end latency across the three models in the shuffled-order experiment.

The two GPU sessions showed that checkpoint 76 had the lowest average model-only and end-to-end latency. However, the difference between models was modest compared with the variation between benchmark sessions.

The preliminary CPU-versus-GPU comparison also suggests that GPU execution can reduce inference latency on this hardware.

Checkpoint selection should consider detection accuracy alongside inference latency. A faster model is not necessarily the better detector if it sacrifices detection quality.

### Benchmark limitations

- Only one laptop hardware configuration was evaluated.
- The benchmark used 50 validation images.
- Two CPU experiments and two three-model GPU sessions were conducted.
- The CPU shuffled-order experiment and GPU experiments were collected in separate sessions.
- The GPU benchmark loaded all three models simultaneously.
- Model execution order was shuffled, but not guaranteed to be perfectly balanced.
- Measurements may be affected by CPU/GPU temperature, background processes, GPU clock behavior, and system power settings.
- Standard deviation across individual image measurements does not represent variation across independent benchmark sessions.
- FPS values are reciprocal-latency estimates rather than independently measured sustained throughput.
- The GPU monitoring data does not independently confirm thermal throttling.
- The validation dataset is not a separate held-out test set.

These results should be treated as preliminary performance measurements rather than statistically conclusive comparisons.

### Reproducibility

The CPU benchmark implementation is located at:

`helmet_detector/scripts/benchmark_inference.py`

The GPU benchmark implementation is located at:

`helmet_detector/scripts/benchmark_inference_gpu.py`

Run the CPU benchmark from the repository root with the CPU project environment activated:

```powershell
python .\helmet_detector\scripts\benchmark_inference.py
```

Run the GPU benchmark from the repository root with the GPU Conda environment activated and its dependencies installed:

```powershell
python .\helmet_detector\scripts\benchmark_inference_gpu.py
```

The benchmark scripts use a fixed random seed, consistent image selection, and consistent warm-up and measurement settings.

Generated CPU benchmark reports are saved locally under:

- `helmet_detector/test_output/performance_benchmark_cpu/`
- `helmet_detector/test_output/performance_benchmark_cpu_round_robin/`

Generated GPU benchmark reports are saved locally under:

- `helmet_detector/test_output/performance_benchmark_gpu/`
- `helmet_detector/test_output/performance_benchmark_gpu_all_models/`
- `helmet_detector/test_output/performance_benchmark_gpu_all_repeat_2/`

Each benchmark output directory contains:

- `benchmark_environment.json`
- `benchmark_summary.json`
- `benchmark_results.csv`

These generated outputs are local benchmark artifacts and are not necessarily included in the Git repository.

### Future optimization

Potential future performance improvements include:

- ONNX export and ONNX Runtime inference.
- TensorFlow Lite conversion and quantization.
- Further GPU performance evaluation under controlled conditions.
- Comparing optimized model latency against the TensorFlow SavedModel baseline.
- Investigating GPU clock and thermal behavior during sustained inference.

Any optimization should be evaluated for both inference speed and detection accuracy.

## Installation

### Requirements

- Python 3.10
- TensorFlow 2.10.0
- Dependencies listed in `requirements.txt`

The project uses `pycocotools-og==2.0.6` for Windows compatibility.

### 1. Clone the repository

```bash
git clone https://github.com/Htet-ThuAung/HelmetDetectionProject.git
cd HelmetDetectionProject
```

### 2. Create and activate a virtual environment

PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Python import path

The TensorFlow Object Detection API contains internal imports that use the top-level `object_detection` package.

From the repository root, configure the path in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\helmet_detector;$PWD"
```

This setting applies to the current PowerShell session.

**Note:** The training dataset and pretrained model assets are not included in the repository. They must be obtained separately before reproducing the full training workflow.

## Running Inference

Run the following commands from the repository root with the virtual environment activated and `PYTHONPATH` configured.

### Single-image inference

```powershell
python -m helmet_detector.scripts.infer_image `
  --image_path "path\to\image.jpg"
```

To specify an output path:

```powershell
python -m helmet_detector.scripts.infer_image `
  --image_path "path\to\image.jpg" `
  --output_path "output\annotated.jpg"
```

### Webcam inference

```powershell
python -m helmet_detector.scripts.infer_webcam
```

Press `q` to exit the webcam window.

The inference scripts use the model and label-map paths configured by the project unless overridden through their command-line arguments.

## Project Structure

```text
HelmetDetectionProject/
├── helmet_detector/
│   ├── config/
│   │   ├── config.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── raw/                 # Local dataset; not tracked
│   │   └── processed/           # Generated TFRecords; not tracked
│   ├── models/
│   │   └── exportedModel/       # Tracked inference model
│   ├── object_detection/        # TensorFlow Object Detection API
│   │   ├── protos/
│   │   └── LICENCE
│   ├── results/
│   │   └── training_plots/
│   ├── scripts/
│   │   ├── benchmark_inference.py
│   │   ├── benchmark_inference_gpu.py
│   │   ├── compare_models.py
│   │   ├── evaluate_checkpoints.py
│   │   ├── evaluate_models.py
│   │   ├── export_model.py
│   │   ├── generate_tfrecord.py
│   │   ├── infer_image.py
│   │   ├── infer_webcam.py
│   │   ├── inspect_annotation.py
│   │   ├── inspect_tensorboard_logs.py
│   │   ├── inspect_validation.py
│   │   ├── plot_training_history.py
│   │   ├── preprocess.py
│   │   ├── show_project_tree.py
│   │   ├── train_model.py
│   │   ├── verify_export.py
│   │   └── visualize_export.py
│   └── test_output/             # Local inference and benchmark outputs
├── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
├── inspect_model.py
├── tensorboard_output.py
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

**Directory notes:**

- `config/` contains project configuration.
- `data/` contains the raw and processed dataset directories.
- `models/` contains model exports, training checkpoints, and pretrained model resources.
- `object_detection/` contains the TensorFlow Object Detection API code.
- `results/` contains generated training and evaluation visualizations.
- `scripts/` contains preprocessing, training, evaluation, export, inference, and benchmarking scripts.
- `test_output/` contains local inference outputs and benchmark reports.

Some directories contain generated or locally maintained files that are excluded from version control.

## What I Learned

Through this project, I worked on:

- Preparing and validating a custom object detection dataset.
- Converting Pascal VOC annotations into TFRecords.
- Fine-tuning an SSD-based object detection model.
- Working with TensorFlow Object Detection API configurations and checkpoints.
- Exporting TensorFlow SavedModels.
- Evaluating detection performance using precision, recall, F1-score, and mAP.
- Comparing model checkpoints using consistent evaluation procedures.
- Building image and webcam inference pipelines.
- Benchmarking CPU and GPU inference latency and analyzing timing variability.
- Managing Python environments, dependencies, and project structure.

## Limitations and Future Improvements

- Expand the dataset with more varied lighting, camera angles, and environmental conditions.
- Add a separate held-out test set.
- Investigate the lower recall and localization consistency of the no-helmet class.
- Conduct additional controlled GPU performance experiments.
- Explore ONNX Runtime and TensorFlow Lite deployment.
- Investigate model optimization and quantization.
- Consider additional safety equipment classes such as vests and masks.

This project is a computer vision portfolio prototype. It has not been validated as a production occupational safety system and should not be used as the sole basis for safety decisions.

## Acknowledgements

- [TensorFlow Models / Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) — for the object detection framework.
- [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) — for pretrained model resources.
- [LabelImg](https://github.com/HumanSignal/labelImg) — for image annotation.

The TensorFlow Object Detection API code included in this repository retains its original Apache License 2.0 notice in `helmet_detector/object_detection/LICENCE`.

## License

The original project code is licensed under the [MIT License](./LICENSE).

Third-party components retain their respective licenses. The TensorFlow Object Detection API is distributed under the Apache License 2.0. See its retained license notice for details.

---

**Author:** Htet Thu Aung  
**GitHub:** [Htet-ThuAung](https://github.com/Htet-ThuAung)