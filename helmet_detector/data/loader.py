import os
import xml.etree.ElementTree as ET

import tensorflow as tf


def load_image(img_path):
    """Load a JPEG image and return it as a TensorFlow tensor."""
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def load_xml_annotations(xml_path):
    """Parse a Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    return tree.getroot()


def get_xml_file_paths(xml_dir):
    """Return sorted XML annotation paths from a directory."""
    xml_dir = os.path.abspath(xml_dir)

    if not os.path.isdir(xml_dir):
        raise FileNotFoundError(
            f"Annotation directory not found: {xml_dir}"
        )

    xml_files = [
        os.path.join(xml_dir, filename)
        for filename in os.listdir(xml_dir)
        if filename.lower().endswith(".xml")
    ]

    return sorted(xml_files)