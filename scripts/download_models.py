"""Download public model files to ignored local storage; no credentials are required."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 1000:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        while block := response.read(1024*1024):
            output.write(block)
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--yolo", action="store_true")
    args = parser.parse_args()
    if not any((args.qwen, args.embedding, args.reranker, args.yolo)):
        parser.error("Select --qwen, --embedding, --reranker or --yolo")
    models = []
    if args.yolo:
        download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", Path("models/yolo11n.pt"))
        print("YOLO weights downloaded", flush=True)
    if args.qwen:
        models.append(("Qwen/Qwen2.5-0.5B-Instruct", "models/qwen2.5-0.5b-instruct"))
    if args.embedding:
        models.append(("BAAI/bge-small-zh-v1.5", "models/bge-small-zh"))
    if args.reranker:
        models.append(("cross-encoder/ms-marco-MiniLM-L-6-v2", "models/ms-marco-reranker"))
    for model_id, local_dir in models:
        from huggingface_hub import snapshot_download
        for attempt in range(8):
            try:
                path = snapshot_download(model_id, local_dir=local_dir,
                    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges.txt", "modules.json", "README.md", "LICENSE"],
                    ignore_patterns=["onnx/*", "openvino/*"], max_workers=2)
                break
            except Exception:
                if attempt == 7:
                    raise
                print(f"Resuming {model_id}, attempt {attempt+2}/8", flush=True)
                time.sleep(2)
        print(json.dumps({"model": model_id, "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
