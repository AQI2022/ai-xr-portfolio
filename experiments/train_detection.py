import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import ai_xr


def main():
    from ultralytics import YOLO
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/taco-mini/dataset.yaml")
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--name", default="taco-mini")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not Path(args.data).is_file():
        raise SystemExit("Run python scripts/prepare_taco.py first")
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=4, workers=0,
                device=args.device, project=str(Path("runs").resolve()), name=args.name, seed=42, deterministic=True,
                plots=False, save=True, exist_ok=True, amp=False, cache=False, verbose=False,
                mosaic=0, close_mosaic=0)
    best = Path(model.trainer.best)
    trained = YOLO(best)
    metrics = trained.val(data=args.data, imgsz=args.imgsz, batch=4, workers=0,
                          device=args.device, project=str(Path("runs").resolve()), name="taco-validation", plots=False, verbose=False)
    sample = next(Path("data/taco-mini/images/val").glob("*.jpg"))
    trained.predict(str(sample), imgsz=args.imgsz, verbose=False, device=args.device)
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        prediction = trained.predict(str(sample), imgsz=args.imgsz, verbose=False, device=args.device)[0]
        latencies.append((time.perf_counter()-t0)*1000)
    report = {"dataset": args.data, "epochs": args.epochs, "image_size": args.imgsz,
              "device": args.device, "platform": platform.system(), "torch": torch.__version__,
              "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
              "median_pipeline_latency_ms": statistics.median(latencies),
              "reciprocal_median_fps": 1000/statistics.median(latencies), "timing_repeats": 10,
              "latencies_ms": latencies, "duration_seconds": time.perf_counter()-start,
              "trained_weights": best.relative_to(Path.cwd()).as_posix(), "classes": trained.names,
              "scope": "Real TACO mini training/validation. Tiny validation set; not production accuracy or full video FPS."}
    Path("evidence").mkdir(exist_ok=True)
    Path(f"evidence/yolo-{args.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
