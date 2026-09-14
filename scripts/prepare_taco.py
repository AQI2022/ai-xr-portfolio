"""Prepare a reproducible small TACO split; do not commit downloaded photographs."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path

from PIL import Image

ANNOTATIONS = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
NAMES = ["bottle", "can", "cup", "bag", "other_litter"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/taco-mini")
    args = parser.parse_args()
    if args.count < 12:
        raise SystemExit("At least 12 images are required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = root/"annotations-original.json"
    if not metadata.exists():
        with urllib.request.urlopen(ANNOTATIONS, timeout=60) as response:
            metadata.write_bytes(response.read())
    data = json.loads(metadata.read_text(encoding="utf-8"))
    categories = {c["id"]: c for c in data["categories"]}
    annotations = {}
    for annotation in data["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    images = [im for im in data["images"] if im["id"] in annotations]
    random.Random(args.seed).shuffle(images)

    def fetch(im):
        url = im.get("flickr_640_url") or im.get("flickr_url")
        if not url:
            return None
        try:
            destination = root/"cache"/f"{im['id']}.jpg"
            if not destination.exists():
                with urllib.request.urlopen(url, timeout=20) as response:
                    raw = response.read(20*1024*1024)
                with Image.open(io.BytesIO(raw)) as image:
                    image = image.convert("RGB")
                    image.thumbnail((640, 640))
                    destination.parent.mkdir(exist_ok=True)
                    image.save(destination, quality=90)
            return im, destination, url
        except Exception:
            return None

    selected = []
    # Preserve shuffled source order even with concurrent downloads.
    for start in range(0, min(len(images), args.count*5), 8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            selected += [r for r in pool.map(fetch, images[start:start+8]) if r is not None]
        if len(selected) >= args.count:
            break
    selected = selected[:args.count]
    if len(selected) < 12:
        raise RuntimeError("Insufficient public images downloaded; dataset was not manufactured")
    split_point = max(1, round(len(selected)*0.75))
    for split, subset in (("train", selected[:split_point]), ("val", selected[split_point:])):
        expected = {str(item[0]["id"]) for item in subset}
        for kind, suffix in (("images", "jpg"), ("labels", "txt")):
            existing = {p.stem for p in (root/kind/split).glob("*."+suffix)}
            if existing and existing != expected:
                raise SystemExit("Existing split differs. Choose a fresh --output directory; no files were deleted.")
    manifest = []
    for index, (im, cached, url) in enumerate(selected):
        split = "train" if index < split_point else "val"
        image_dir, label_dir = root/"images"/split, root/"labels"/split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        raw = cached.read_bytes()
        (image_dir/f"{im['id']}.jpg").write_bytes(raw)
        rows = []
        for ann in annotations[im["id"]]:
            category = categories[ann["category_id"]]
            name = category["supercategory"].lower()
            cls = 0 if name == "bottle" else 1 if name == "can" else 2 if name == "cup" else 3 if "bag" in name else 4
            x, y, w, h = ann["bbox"]
            iw, ih = im["width"], im["height"]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(iw, x+w), min(ih, y+h)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append(f"{cls} {(x1+x2)/(2*iw):.7f} {(y1+y2)/(2*ih):.7f} {(x2-x1)/iw:.7f} {(y2-y1)/ih:.7f}")
        (label_dir/f"{im['id']}.txt").write_text("\n".join(rows), encoding="utf-8")
        manifest.append({"image_id": im["id"], "split": split, "source_url": url,
                         "license_id": im.get("license"), "sha256": hashlib.sha256(raw).hexdigest(),
                         "boxes": len(rows)})
    yaml = "path: " + root.as_posix() + "\ntrain: images/train\nval: images/val\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i,name in enumerate(NAMES))
    (root/"dataset.yaml").write_text(yaml, encoding="utf-8")
    report = {"dataset": "TACO mini", "source": "https://github.com/pedropro/TACO",
              "annotation_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
              "seed": args.seed, "image_count": len(selected), "train_images": split_point,
              "validation_images": len(selected)-split_point, "class_names": NAMES,
              "licenses": data.get("licenses", []), "manifest": manifest,
              "scope": "Small real litter dataset, image-disjoint split. No independent test set or generalization claim."}
    Path("evidence").mkdir(exist_ok=True)
    Path("evidence/taco-split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(selected), "train": split_point, "validation": len(selected)-split_point}), flush=True)


if __name__ == "__main__":
    main()
