#  Helmet Detection using TensorFlow Object Detection API

This project aims to detect whether a person is wearing a helmet or not using a real-time object detection model built with TensorFlow's Object Detection API. The model is trained on a **custom dataset** annotated manually using **LabelImg**, and deployed for both image and webcam inference.

---


## Features

-  Real-time detection using webcam
-  Lightweight SSD MobileNetV2 backbone
-  Pascal VOC-style annotation support
-  Custom training on a helmet dataset
-  TFRecord data pipeline
-  Live video output with bounding boxes and labels

---


## Project Structure
```HelmetDetectionProject/
├── helmet_detector/
│   ├── config/
│   │   └── config.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── raw/
│   │   │   ├── train/
│   │   │   │   ├── images/
│   │   │   │   └── annotations/
│   │   │   ├── val/
│   │   │   │   ├── images/
│   │   │   │   └── annotations/
│   │   │   └── label_map.pbtxt
│   │   └── processed/
│   │       ├── train/
│   │       │   ├── images/
│   │       │   └── annotations/
│   │       ├── val/
│   │       │   ├── images/
│   │       │   └── annotations/
│   │       └── label_map.pbtxt
│   ├── models/
│   │   └── helmet_detection_ssd_mnet_v2/
│   │       ├── tf_model_zoo/
│   │       └── exportedModel/
│   ├── object_detection/
│   ├── scripts/
│   │   ├── preprocess.py
│   │   ├── generate_tfrecord.py
│   │   ├── train_model.py
│   │   ├── export_model.py
│   │   ├── infer_image.py
│   │   └── infer_webcam.py
│   └── test_output/
│       ├── test_images/
│       └── test_video/
├── README.md
├── requirements.txt
└── LICENSE
```


---


## Model

- Architecture: **SSD with MobileNet V2**
- Framework: **TensorFlow Object Detection API**
- Config: Fine-tuned pipeline based on TensorFlow Model Zoo

---


## Dataset

- **Images**:
  - Training: `572`
  - Validation: `236`
- **Annotations**: Collected using [LabelImg](https://github.com/tzutalin/labelImg) in Pascal VOC (`.xml`) format

- **Classes**:
  - `Helmet`
  - `No-Helmet`

- **Preprocessing Steps**:
  - Converted VOC XML to CSV
  - Generated **TFRecord** files for training and validation

---


## Features

- Real-time webcam detection
- Single-image inference
- Modular script-based training and evaluation
- Export model to `SavedModel` format
- Post-inference testing with videos/images

---


## How to Run

### 1. Install dependencies

```bash
# Clone the repo
git clone https://github.com/yourusername/HelmetDetectionProject.git

# Install requirements
pip install -r requirements.txt
```

### 2. Preprocess and generate TFRecord

`python helmet_detector/scripts/preprocess.py`
`python helmet_detector/scripts/generate_tfrecord.py`

### 3. Train the model
`python helmet_detector/scripts/train_model.py`

### 4. Export trained model
`python helmet_detector/scripts/export_model.py`

### 5. Run inference
- **On single image**
`python helmet_detector/scripts/infer_image.py --image_path path/to/image.jpg`

- **On webcam**
`python helmet_detector/scripts/infer_webcam.py`





### What I Learned

- Collecting and annotating a custom dataset using LabelImg

- Preprocessing XML annotations into TFRecord format

- Training a custom object detector using TensorFlow’s Object Detection API

- Tweaking the SSD MobileNetV2 pipeline config for improved results

- Handling path resolution and training scripts across multiple directories

- Deploying models for image and real-time webcam detection

### Future Improvements
- Add more data(better quality & under various conditions) and improve accuracy

- Implement multi-class detection (e.g., helmet, mask, vest)

- Deploy as a web app or Android app


## Sample Results

### Tensorboard Visualization
```bash
tensorboard --logdir=helmet_detector/models/helmet_detection_ssd_mnet_v2/logs
```

### Evaluate the model while training or after training
```
python helmet_detector/object_detection/model_main_tf2.py ^
  --model_dir=helmet_detector/models/helmet_detection_ssd_mnet_v2 ^
  --pipeline_config_path=helmet_detector/models/helmet_detection_ssd_mnet_v2/pipeline.config ^
  --checkpoint_dir=helmet_detector/models/helmet_detection_ssd_mnet_v2 ^
  --eval_timeout=0
```

### Detection on Test Images
Here are a few detection results on unseen images:

<p float="left">
  <img src="helmet_detector/test_output/test_images/helmet_detected_image.jpg" width="30%" />
  <img src="helmet_detector/test_output/test_images/no_helmet_detected_image.jpg" width="30%" />
  <img src="helmet_detector/test_output/test_images/sample_3.jpg" width="30%" />
  <img src="helmet_detector/test_output/test_images/sample_4.jpg" width="30%" />
  
</p>

> You can find the full output video at:  
> `helmet_detector/test_output/test_video/output_video.mp4`

### Acknowledgements

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) – for the powerful detection framework.
- [LabelImg](https://github.com/heartexlabs/labelImg) – for easy and effective image annotation.


### License
This project is licensed under the [MIT License](./LICENSE).



## Motivation

Many industrial or traffic accidents occur due to negligence in wearing helmets, leading to preventable injuries and fatalities. Ensuring helmet compliance manually can be time-consuming, inconsistent, or impractical in large or busy environments. This project aims to automate helmet detection using computer vision and deep learning, offering a scalable and real-time solution to enhance safety in hazardous zones.
