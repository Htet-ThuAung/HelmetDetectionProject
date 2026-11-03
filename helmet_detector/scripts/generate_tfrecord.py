
import argparse
import io
import os
import glob
import xml.etree.ElementTree as ET

import pandas as pd
import tensorflow as tf
from PIL import Image

from helmet_detector.config import config
from helmet_detector.object_detection.utils import dataset_util, label_map_util


def get_split_paths(split_name):
    """Resolve image, annotation, TFRecord and CSV paths for a dataset split."""

    if split_name == "train":
        image_dir = config.RAW_IMAGES_DIR
        annotation_dir = config.RAW_ANNOTATIONS_DIR
        record_path = config.TRAIN_TFRECORD_PATH
        csv_path = config.TRAIN_ANNOTATION_PATH

    else:
        image_dir = config.VAL_IMAGES_DIR
        annotation_dir = config.VAL_ANNOTATIONS_DIR
        record_path = os.path.join(config.TFRECORD_DIR, "val.record")
        csv_path = os.path.join(config.TFRECORD_DIR, "val_annotations.csv")

    return image_dir, annotation_dir, record_path, csv_path


def load_label_map(label_map_path):
    """Load class names and their integer IDs."""
    if not os.path.isfile(label_map_path):
        raise FileNotFoundError(
            f"Label map not found: {label_map_path}"
        )

    label_map = label_map_util.load_labelmap(label_map_path)
    return label_map_util.get_label_map_dict(label_map)


def xml_to_dataframe(annotation_dir):
    """Convert Pascal VOC XML annotations into a DataFrame."""

    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError(
            f"Annotation directory not found: {annotation_dir}"
        )

    xml_files = sorted(
        glob.glob(os.path.join(annotation_dir, "*.xml"))
    )

    if not xml_files:
        raise ValueError(
            f"No XML annotation files found in: {annotation_dir}"
        )

    records = []

    for xml_file in xml_files:
        root = ET.parse(xml_file).getroot()

        filename = root.findtext("filename")
        size = root.find("size")

        if not filename or size is None:
            raise ValueError(
                f"Missing filename or image size in {xml_file}"
            )

        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        for obj in root.findall("object"):
            box = obj.find("bndbox")

            if box is None:
                raise ValueError(
                    f"Missing bounding box in {xml_file}"
                )

            records.append({
                "filename": filename,
                "width": width,
                "height": height,
                "class": obj.findtext("name"),
                "xmin": int(box.findtext("xmin")),
                "ymin": int(box.findtext("ymin")),
                "xmax": int(box.findtext("xmax")),
                "ymax": int(box.findtext("ymax")),
            })

    columns = [
        "filename", "width", "height", "class",
        "xmin", "ymin", "xmax", "ymax"
    ]

    return pd.DataFrame(records, columns=columns)


def create_tf_example(group, image_dir, label_map_dict):
    """Create a TensorFlow Example from one image and its annotations."""

    image_path = os.path.join(image_dir, group.filename)

    if not os.path.isfile(image_path):
        print(f"[WARNING] Image not found: {image_path}")
        return None

    with tf.io.gfile.GFile(image_path, "rb") as image_file:
        encoded_image = image_file.read()

    with Image.open(io.BytesIO(encoded_image)) as image:
        width, height = image.size
        image_format = image.format.lower().encode("utf8")

    filename = group.filename.encode("utf8")

    xmins, xmaxs = [], []
    ymins, ymaxs = [], []
    classes_text, classes = [], []

    for _, row in group.object.iterrows():
        class_name = row["class"]

        if class_name not in label_map_dict:
            print(
                f"[WARNING] Unknown class '{class_name}' "
                f"in {group.filename}"
            )
            continue

        # Pascal VOC coordinates are 1-based.
        # Convert minimum coordinates to zero-based coordinates.
        xmin = row["xmin"] - 1
        ymin = row["ymin"] - 1
        xmax = row["xmax"]
        ymax = row["ymax"]

        # Validate bounding box coordinates.
        if not (
            0 <= xmin < xmax <= width
            and 0 <= ymin < ymax <= height
        ):
            print(
                f"[WARNING] Invalid bounding box in "
                f"{group.filename}: {xmin}, {ymin}, {xmax}, {ymax}"
            )
            continue

        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)

        classes_text.append(class_name.encode("utf8"))
        classes.append(label_map_dict[class_name])

    if not classes:
        print(
            f"[WARNING] Skipping {group.filename}: "
            "no valid annotations."
        )
        return None

    features = {
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(filename),
        "image/source_id": dataset_util.bytes_feature(filename),
        "image/encoded": dataset_util.bytes_feature(encoded_image),
        "image/format": dataset_util.bytes_feature(image_format),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
        "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_util.int64_list_feature(classes),
    }

    return tf.train.Example(
        features=tf.train.Features(feature=features)
    )


def generate_tfrecord(split_name):
    """Generate a TFRecord and annotation CSV for a dataset split."""

    image_dir, annotation_dir, record_path, csv_path = (
        get_split_paths(split_name)
    )

    print(f"\n{'=' * 50}")
    print(f"Generating {split_name.upper()} TFRecord")
    print(f"{'=' * 50}")
    print(f"Images:      {image_dir}")
    print(f"Annotations: {annotation_dir}")
    print(f"TFRecord:    {record_path}")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    label_map_dict = load_label_map(config.LABEL_MAP_PATH)
    examples = xml_to_dataframe(annotation_dir)

    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    grouped = examples.groupby("filename", sort=True)

    written_images = 0
    skipped_images = 0
    written_objects = 0

    with tf.io.TFRecordWriter(record_path) as writer:
        for filename, object_group in grouped:
            group = type(
                "ImageGroup",
                (),
                {
                    "filename": filename,
                    "object": object_group,
                }
            )()

            tf_example = create_tf_example(
                group,
                image_dir,
                label_map_dict
            )

            if tf_example is None:
                skipped_images += 1
                continue

            writer.write(tf_example.SerializeToString())
            written_images += 1

            written_objects += len(
                tf_example.features.feature[
                    "image/object/class/label"
                ].int64_list.value
            )

    examples.to_csv(csv_path, index=False)

    print("\nGeneration completed.")
    print(f"Images written:  {written_images}")
    print(f"Images skipped:  {skipped_images}")
    print(f"Objects written: {written_objects}")
    print(f"TFRecord:        {record_path}")
    print(f"CSV:             {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TensorFlow TFRecords from Pascal VOC datasets."
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        required=True,
        help="Dataset split to convert."
    )

    args = parser.parse_args()
    generate_tfrecord(args.split)


if __name__ == "__main__":
    main()
