from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    "__pycache__",
    ".git",
    "train",
    "val",
    "raw",
    "object_detection",
    "test_images",
    "test_output",
    "logs",
    "checkpoint",
    "best_checkpoint_76",
    "tf_model_zoo",
    "helmet_detection_1",
}

SKIP_FILES = {
    ".DS_Store",
}

def show_tree(folder, prefix=""):
    entries = sorted(
        [
            path for path in folder.iterdir()
            if path.name not in SKIP_FILES
            and not (path.is_dir() and path.name in SKIP_DIRS)
            and not path.name.startswith("ckpt-")
            and not path.name.startswith("events.out.tfevents")
        ],
        key=lambda path: (path.is_file(), path.name.lower())
    )

    for index, path in enumerate(entries):
        last = index == len(entries) - 1
        print(prefix + ("└── " if last else "├── ") + path.name)

        if path.is_dir():
            show_tree(
                path,
                prefix + ("    " if last else "│   ")
            )

print(ROOT.name + "/")
show_tree(ROOT)