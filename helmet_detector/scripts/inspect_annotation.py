
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_DIR = ROOT / "data" / "raw" / "val" / "annotations"
LABEL_MAP_PATH = ROOT / "data" / "processed" / "label_map" / "label_map.pbtxt"

# Inspect one annotation
xml_file = sorted(ANNOTATION_DIR.glob("*.xml"))[0]
root = ET.parse(xml_file).getroot()

print("Sample annotation:", xml_file.name)
print("Image filename:", root.findtext("filename"))

size = root.find("size")
if size is not None:
    print(
        "Image dimensions:",
        size.findtext("width"),
        "x",
        size.findtext("height"),
    )

print("\nGround-truth objects:")

for obj in root.findall("object"):
    name = obj.findtext("name")
    box = obj.find("bndbox")

    if box is not None:
        coordinates = {
            key: box.findtext(key)
            for key in ("xmin", "ymin", "xmax", "ymax")
        }
        print(name, coordinates)

print("\nLabel map:")
print(LABEL_MAP_PATH.read_text(encoding="utf-8"))
