import os
import tensorflow as tf
from helmet_detector.data.loader import load_image, load_xml_annotations, get_xml_file_paths
import xml.etree.ElementTree as ET
import argparse
from helmet_detector.config import config  # ← import the shared config


def letterbox_image(image, target_size):
    """Resize image with unchanged aspect ratio using padding."""
    original_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    ratio = tf.minimum(target_size[1] / original_shape[1],
                       target_size[0] / original_shape[0])

    new_shape = tf.cast(original_shape * ratio, tf.int32)
    image_resized = tf.image.resize(image, new_shape)

    pad_y = target_size[0] - new_shape[0]
    pad_x = target_size[1] - new_shape[1]

    # Cast to int32 and return values for further processing
    image_padded = tf.image.pad_to_bounding_box(
        image_resized,
        tf.cast(pad_y // 2, tf.int32),
        tf.cast(pad_x // 2, tf.int32),
        target_size[0],
        target_size[1]
    )
    return image_padded, ratio, int(pad_x), int(pad_y)


def update_xml_bounding_boxes(root, ratio, pad_x, pad_y):
    """Update bounding boxes in XML annotations based on scaling."""
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        for tag in ["xmin", "ymin", "xmax", "ymax"]:
            coord = int(bndbox.find(tag).text)
            if tag in ["xmin", "xmax"]:
                coord = int(coord * ratio + pad_x // 2)
            else:
                coord = int(coord * ratio + pad_y // 2)
            bndbox.find(tag).text = str(coord)

def parse_annotations(xml_dir, output_dir, img_dir, target_size):
    """Parses xml annotations and preprocesses images using Tensorflow."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    try:
        xml_files = get_xml_file_paths(xml_dir)
    except FileNotFoundError as e:
        print(e)
        return

    for xml_file in xml_files:
        try:
            root = load_xml_annotations(xml_file)
            filename = root.find("filename").text
            img_path = os.path.join(img_dir, filename)

            if not tf.io.gfile.exists(img_path):
                print(f"Image {filename} not found! Skipping.")
                continue

            image = load_image(img_path)
            image_padded, ratio, pad_x, pad_y = letterbox_image(image, target_size)
            image_padded = tf.image.convert_image_dtype(image_padded, tf.float32)

            update_xml_bounding_boxes(root, ratio, pad_x, pad_y)

            tf.keras.utils.save_img(os.path.join(output_dir, "images", filename), image_padded)
            tree = ET.ElementTree(root)
            tree.write(os.path.join(output_dir, "annotations", os.path.basename(xml_file)))

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

    print("Preprocessing Complete!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images and annotations!!")

    parser.add_argument("-x", "--xml_dir", type=str, default=config.RAW_ANNOTATIONS_DIR,
                        help="Path to xml annotations.")
    parser.add_argument("-i", "--img_dir", type=str, default=config.RAW_IMAGES_DIR,
                        help="Path to images.")
    parser.add_argument("-o", "--output_dir", type=str, default=config.PROCESSED_DATA_DIR,
                        help="Path to save preprocessed data.")
    parser.add_argument("-s", "--size", nargs=2, type=int, default=config.IMAGE_SIZE,
                        help="Target image size (width, height).")

    args = parser.parse_args()
    parse_annotations(args.xml_dir, args.output_dir, args.img_dir, tuple(args.size))
