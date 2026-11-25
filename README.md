# image-dataset-auditor

Quality audit for image classification / detection datasets before training:
exact duplicates, near-duplicates (perceptual hash), train/val/test split
leakage, images without labels, and class imbalance.

Pure Python 3.10+, zero required dependencies (PGM P2/P5 native; register a
Pillow-based loader for PNG/JPEG in real projects, see below).

## Usage

```python
from imgaudit import audit_directory

findings = audit_directory("path/to/dataset")   # split dirs: train/ val/ test/
for f in findings:
    print(f.kind, "-", f.detail, *f.paths, sep="\n  ")
```

With Pillow for PNG/JPEG:

```python
from PIL import Image
from imgaudit import hashes

def pillow_loader(path):
    im = Image.open(path).convert("L")
    return im.width, im.height, im.tobytes()

hashes.register_loader(pillow_loader)
```

## What it reports

| kind | meaning |
|------|---------|
| `exact_dup` | byte-identical files (sha256) |
| `near_dup` | perceptual dHash hamming distance <= 5 |
| `leakage` | identical image present in more than one split |
| `missing_image` | image with no matching label (.txt beside it, or CSV row) |
| `class_imbalance` | largest class > 10x the smallest |

Labels: YOLO-style `<class> cx cy w h` text files beside each image, or a
CSV with `image,class` rows.

## Honest limits

- dHash is sensitive to *gradients* and cheap to compute, but weak against
  crops/rotations; it catches resized/copy-pasted near-dups, not augmented
  ones. Smooth synthetic gradients hash degenerately (documented in tests).
- Near-dup comparison is O(n^2) over in-memory hashes: fine to ~10k images,
  needs a blocking strategy beyond that.
- Native decoding covers PGM only; other formats raise `ValueError` unless a
  loader is registered (exact-duplicate detection still works, it is
  byte-based).
- Label parsing assumes one class token per line/row; multi-object YOLO
  files count each row toward class balance.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

7 tests generating deterministic PGM fixtures: hash stability, dHash
discrimination, exact-dup + leakage detection, far-image non-flagging,
missing labels, and imbalance ratio reporting.
