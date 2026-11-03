
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data" / "raw" / "val" / "images"
ANNOTATION_DIR = ROOT / "data" / "raw" / "val" / "annotations"
image_extensions = {".jpg", ".jpeg", ".png"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

annotations = list(ANNOTATION_DIR.glob("*.xml"))

image_stems = {p.stem for p in images}
annotation_stems = {p.stem for p in annotations}

print("Validation dataset inspection")
print("-" * 40)
print("Images:", len(images))
print("Annotations:", len(annotations))
print("Images without annotations:", len(image_stems - annotation_stems))
print("Annotations without images:", len(annotation_stems - image_stems))

classes = {}

for xml_file in annotations:
    root = ET.parse(xml_file).getroot()

    for obj in root.findall("object"):
        name = obj.findtext("name", default="unknown")
        classes[name] = classes.get(name, 0) + 1

print("\nAnnotation classes:")
for name, count in sorted(classes.items()):
    print(f"{name}: {count}")

print("\nSample image files:")
for path in images[:5]:
    print(path.name)

print("\nSample annotation files:")
for path in annotations[:5]:
    print(path.name)
