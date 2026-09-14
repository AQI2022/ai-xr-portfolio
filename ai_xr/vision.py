import io
import threading
import time
from pathlib import Path

from PIL import Image


class Detector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.lock = threading.RLock()

    def detect(self, content, confidence=0.25):
        from ultralytics import YOLO
        with Image.open(io.BytesIO(content)) as opened:
            if opened.width * opened.height > 20_000_000:
                raise ValueError("Image exceeds 20 megapixels")
            image = opened.convert("RGB")
        with self.lock:
            if not Path(self.model_path).is_file():
                raise FileNotFoundError("YOLO weights missing; run scripts/download_models.py --yolo")
            if self.model is None:
                self.model = YOLO(self.model_path)
                self.model.predict(Image.new("RGB", (320, 320)), imgsz=320, verbose=False)
            started = time.perf_counter()
            result = self.model.predict(image, conf=confidence, imgsz=640, verbose=False)[0]
            elapsed = time.perf_counter() - started
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append({"label": result.names[class_id], "class_id": class_id,
                               "confidence": round(float(box.conf[0]), 4),
                               "xyxy": [round(float(x), 2) for x in box.xyxy[0]],
                               "xyxyn": [round(float(x), 5) for x in box.xyxyn[0]]})
        return {"detections": detections, "width": image.width, "height": image.height,
                "latency_ms": round(elapsed*1000, 2), "single_image_fps": round(1/elapsed, 2),
                "model": Path(self.model_path).name,
                "scope": "Model classes depend on loaded weights; COCO weights are not a trained waste taxonomy."}


async def describe_image(content, provider, question="描述图中可见物体及其空间关系。"):
    import base64
    with Image.open(io.BytesIO(content)) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        output = io.BytesIO()
        image.save(output, format="JPEG")
    if provider.mode != "openai-compatible":
        raise ValueError("VLM requires a configured image-capable compatible endpoint")
    data_url = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()
    response = await provider.chat([{"role": "user", "content": [
        {"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": data_url}}]}])
    return {"description": response.get("content"), "mode": provider.mode}
