import os
import tensorflow as tf
import xml.etree.ElementTree as ET

def load_image(img_path):
    """Load image from disk and return as a tensor."""
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_xml_annotations(xml_path):
    """Parse XML annotations and return the root of the XML."""
    tree = ET.parse(xml_path)
    return tree.getroot()

def get_xml_file_paths(xml_dir):
    """Returns a list of XML file paths in the specified directory."""
    # Convert relative path to absolute path
    xml_dir = os.path.abspath(xml_dir)
    print(f"Looking for XML files in: {xml_dir}")
    
    # Ensure the directory exists
    if not os.path.exists(xml_dir):
        print(f"Error: The directory {xml_dir} does not exist!")
        return []
    
    # List XML files
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} XML files in {xml_dir}")
    return xml_files

# Update this path to match your actual file structure
xml_dir = "D:/Projects/HelmetDetectionProject/my_package/data/raw/train/annotations"

# Call the function to get XML files
xml_files = get_xml_file_paths(xml_dir)

# If needed, you can print the xml_files to check the file paths
if xml_files:
    print(f"XML files found: {xml_files}")
else:
    print("No XML files found or directory does not exist.")
