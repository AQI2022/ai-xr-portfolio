import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_xr.api import create_app
from ai_xr.config import Settings


def main():
    sample = next(Path("data/taco-mini/images/val").glob("*.jpg"))
    results = []
    for weights in ("models/yolo11n.pt", "runs/taco-40/weights/best.pt"):
        with TestClient(create_app(Settings(yolo_model=weights))) as client:
            response = client.post("/vision/detect", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")})
            response.raise_for_status()
            results.append({"weights": weights, "sample_id": sample.stem, "result": response.json()})
    Path("evidence/vision-api.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{"weights": r["weights"], "detections": len(r["result"]["detections"])} for r in results]))


if __name__ == "__main__":
    main()
