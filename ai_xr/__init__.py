"""AI and XR portfolio services."""
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("data/ultralytics").resolve()))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(Path("models/cache").resolve()))
__version__ = "0.1.0"
