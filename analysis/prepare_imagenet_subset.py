"""Build the benchmark's labelled evaluation set from a local copy of ImageNet-1k validation.

ImageNet cannot be redistributed, so each researcher builds the subset locally. The sample is
drawn with a fixed seed, so the same arguments always yield the same file list (record the
seed and the printed SHA-256 of the manifest in the paper).

Preprocessing follows the standard MobileNet evaluation: resize the shorter side to 256 px,
center-crop 224x224, save losslessly as PNG. The benchmark then feeds the 224x224 image as-is.

Two input layouts are supported:
  1. Kaggle ILSVRC layout:  --val-dir .../ILSVRC/Data/CLS-LOC/val
                            --solution LOC_val_solution.csv --synsets LOC_synset_mapping.txt
  2. Class-folder layout:   --val-dir val/  (one sub-folder per WNID, e.g. val/n01440764/*.JPEG,
                            or per class index as in ImageNetV2, e.g. val/0/*.jpeg)

Labels are 0-based indices into the sorted WNID list, which is the class order MobileNet emits.

Example:
  python analysis/prepare_imagenet_subset.py --val-dir D:/imagenet/val --n 1000 --seed 20261017
"""
import argparse
import csv
import hashlib
import random
from pathlib import Path

from PIL import Image


def load_samples(args):
    val_dir = Path(args.val_dir)
    if args.solution:
        wnids = sorted(line.split()[0] for line in Path(args.synsets).read_text().splitlines() if line.strip())
        index = {w: i for i, w in enumerate(wnids)}
        samples = []
        with open(args.solution, newline="") as f:
            for row in csv.DictReader(f):
                wnid = row["PredictionString"].split()[0]
                samples.append((val_dir / f"{row['ImageId']}.JPEG", index[wnid]))
    else:
        wnids = sorted(p.name for p in val_dir.iterdir() if p.is_dir())
        if all(w.isdigit() for w in wnids):
            # ImageNetV2 layout: folders are already the 0-based class indices ("0".."999").
            samples = [(img, int(w)) for w in wnids for img in sorted((val_dir / w).iterdir()) if img.is_file()]
        else:
            samples = [(img, i) for i, w in enumerate(wnids) for img in sorted((val_dir / w).iterdir()) if img.is_file()]
    if len(wnids) != 1000:
        raise SystemExit(f"Expected 1000 classes, found {len(wnids)}")
    return sorted(samples)


def preprocess(src, dst, resize=256, crop=224):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = resize / min(w, h)
        im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        w, h = im.size
        left, top = (w - crop) // 2, (h - crop) // 2
        im.crop((left, top, left + crop, top + crop)).save(dst, format="PNG")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--solution", help="LOC_val_solution.csv (Kaggle layout)")
    ap.add_argument("--synsets", help="LOC_synset_mapping.txt (Kaggle layout)")
    ap.add_argument("--n", type=int, default=1000, help="number of images to sample")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", default="public/dataset")
    args = ap.parse_args()
    if bool(args.solution) != bool(args.synsets):
        ap.error("--solution and --synsets must be given together")

    samples = load_samples(args)
    chosen = random.Random(args.seed).sample(samples, min(args.n, len(samples)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (src, label) in enumerate(chosen):
        name = f"{src.stem}.png"
        preprocess(src, out / name)
        rows.append((name, label))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(chosen)}")

    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        w.writerows(rows)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    print(f"Wrote {len(rows)} images to {out}\nmanifest sha256: {digest}")


if __name__ == "__main__":
    main()
