import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple
from helmet_detector.config import config  # Importing your config file

# Use the paths from config.py
XML_DIR = config.RAW_ANNOTATIONS_DIR  # XML annotations folder
IMAGE_DIR = config.RAW_IMAGES_DIR  # Raw image folder
LABELS_PATH = config.LABEL_PATH  # Label map path
OUTPUT_PATH = config.TRAIN_TFRECORD_PATH  # Path to save the TFRecord file
CSV_PATH = config.TRAIN_ANNOTATION_PATH

label_map = label_map_util.load_labelmap(LABELS_PATH)
label_map_dict = label_map_util.get_label_map_dict(label_map)


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            value = (
                root.find("filename").text,
                int(root.find("size")[0].text),
                int(root.find("size")[1].text),
                member.find("name").text,  # Object class
                int(member.find("bndbox/xmin").text),
                int(member.find("bndbox/ymin").text),
                int(member.find("bndbox/xmax").text),
                int(member.find("bndbox/ymax").text),
            )
            xml_list.append(value)

    column_names = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    return xml_df


def class_text_to_int(row_label):
    if row_label in label_map_dict:
        return label_map_dict[row_label]
    else:
        print(f"Label '{row_label}' not found in label_map.pbtxt")
        return None  # Or raise an error


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    image_path = os.path.join(IMAGE_DIR, group.filename)
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return None  # Skip this example

    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b"jpg"
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    for _, row in group.object.iterrows():
        if row["class"] not in label_map_dict:
            print(f"WARNING: Class '{row['class']}' not found in label map!")
            continue  # Skip this annotation

        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(row["class"]))

    if len(classes) == 0:
        print(f"Skipping {group.filename} (No valid annotations)")
        return None  # Skip if no valid objects

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(filename),
        "image/source_id": dataset_util.bytes_feature(filename),
        "image/encoded": dataset_util.bytes_feature(encoded_jpg),
        "image/format": dataset_util.bytes_feature(image_format),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
        "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    writer = tf.io.TFRecordWriter(OUTPUT_PATH)
    path = os.path.join(IMAGE_DIR)
    examples = xml_to_csv(XML_DIR)
    grouped = split(examples, "filename")
    for group in grouped:
        tf_example = create_tf_example(group, path)
        if tf_example:
            writer.write(tf_example.SerializeToString())
        else:
            print(f"Skipping {group.filename}, no valid data found!")
    writer.close()
    print("Successfully created the TFRecord file: {}".format(OUTPUT_PATH))
    if CSV_PATH is not None:
        examples.to_csv(CSV_PATH, index=None)
        print("Successfully created the CSV file: {}".format(CSV_PATH))


if __name__ == "__main__":
    main()
