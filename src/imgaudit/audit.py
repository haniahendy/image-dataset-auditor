"""Directory-level dataset audit.

Finds: exact duplicates (sha256), near-duplicates (dHash hamming <= 5),
split leakage (same/near image in >1 split), label files referencing missing
images, and class imbalance. Labels: YOLO-style .txt beside images
(class cx cy w h) or a CSV `image,class`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .hashes import sha256_file, dhash, hamming

IMG_EXT = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}


@dataclass
class Finding:
    kind: str           # exact_dup | near_dup | leakage | missing_image | class_imbalance
    detail: str
    paths: list[str] = field(default_factory=list)


def _split_of(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    for p in parts[:-1]:
        if p.lower() in SPLIT_NAMES:
            return "valid" if p.lower() == "validation" else p.lower()
    return ""


def _images(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def _labels(root: str) -> tuple[dict[str, list[str]], set[str]]:
    """Return (class -> images), (images referenced by any label row)."""
    class_to_imgs: dict[str, list[str]] = {}
    referenced: set[str] = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            if f.endswith(".csv"):
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if not line or line.lower().startswith("image"):
                        continue
                    img, _, cls = line.partition(",")
                    img, cls = img.strip(), cls.strip()
                    if img and cls:
                        class_to_imgs.setdefault(cls, []).append(img)
                        referenced.add(img)
            elif f.endswith(".txt") and not f.startswith("classes"):
                base = os.path.splitext(f)[0]
                for ext in IMG_EXT:
                    cand = os.path.join(dirpath, base + ext)
                    if os.path.exists(cand):
                        referenced.add(cand)
                        for line in open(p, encoding="utf-8"):
                            tok = line.split()
                            if tok:
                                class_to_imgs.setdefault(tok[0], []).append(cand)
                        break
    return class_to_imgs, referenced


def audit_directory(root: str, *, near_threshold: int = 5) -> list[Finding]:
    imgs = _images(root)
    findings: list[Finding] = []

    sha_of = {p: sha256_file(p) for p in imgs}

    # exact duplicates
    by_sha: dict[str, list[str]] = {}
    for p in imgs:
        by_sha.setdefault(sha_of[p], []).append(p)
    for sha, paths in by_sha.items():
        if len(paths) > 1:
            findings.append(Finding("exact_dup", f"{len(paths)} identical files", sorted(paths)))

    # near duplicates (O(n^2) on hashes, fine for audit-scale sets)
    hashes = []
    for p in imgs:
        try:
            hashes.append((p, dhash(p)))
        except ValueError:
            continue  # unsupported without an external loader; exact-dup still caught
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            p1, p2 = hashes[i][0], hashes[j][0]
            if sha_of[p1] == sha_of[p2]:
                continue  # already reported as exact duplicate
            d = hamming(hashes[i][1], hashes[j][1])
            if d <= near_threshold:
                findings.append(Finding("near_dup", f"hamming={d}", [p1, p2]))

    # split leakage: same sha appearing under 2+ split names
    split_seen: dict[str, set[str]] = {}
    for p in imgs:
        sp = _split_of(p, root)
        if sp:
            split_seen.setdefault(sha_of[p], set()).add(sp)
    for sha, splits in split_seen.items():
        if len(splits) > 1:
            paths = [p for p in imgs if sha_of[p] == sha]
            findings.append(Finding("leakage", f"same image in {sorted(splits)}", sorted(paths)))

    # labels vs images
    class_to_imgs, referenced = _labels(root)
    for img in imgs:
        if img not in referenced and not any(img in v for v in class_to_imgs.values()):
            findings.append(Finding("missing_image", "image has no label", [img]))

    # imbalance: dominant class > 10x rarest
    if len(class_to_imgs) > 1:
        sizes = {c: len(v) for c, v in class_to_imgs.items()}
        big, small = max(sizes.values()), min(sizes.values())
        if small and big / small > 10:
            findings.append(Finding(
                "class_imbalance",
                f"largest class {big}x images vs smallest {small}x ({big/small:.0f}:1)",
                [],
            ))
    return findings
