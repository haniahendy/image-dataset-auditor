import os
import tempfile
import unittest

from imgaudit import sha256_file, dhash, hamming, audit_directory


def pgm_ascii(w, h, base=0, step=1):
    """Deterministic ASCII PGM with a gentle horizontal-ish gradient."""
    vals = " ".join(str((base + (i * step)) % 256) for i in range(w * h))
    return f"P2\n{w} {h}\n255\n{vals}\n"


def pgm_ramp_rev(w, h):
    """Horizontal gradient decreasing left to right (opposite dHash of ramp)."""
    vals = " ".join(str(255 - (x * 255) // max(1, w - 1)) for y in range(h) for x in range(w))
    return f"P2\n{w} {h}\n255\n{vals}\n"


def pgm_ramp(w, h):
    vals = " ".join(str((x * 255) // max(1, w - 1)) for y in range(h) for x in range(w))
    return f"P2\n{w} {h}\n255\n{vals}\n"


class TestHashes(unittest.TestCase):
    def test_sha256_stable(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pgm", delete=False) as f:
            f.write(pgm_ascii(9, 8))
            path = f.name
        self.assertEqual(sha256_file(path), sha256_file(path))
        os.unlink(path)

    def test_dhash_deterministic_and_symmetric_hamming(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pgm", delete=False) as f:
            f.write(pgm_ramp(18, 16))
            path = f.name
        a, b = dhash(path), dhash(path)
        self.assertEqual(a, b)
        self.assertEqual(hamming(a, b), 0)
        os.unlink(path)

    def test_dhash_differs_for_different_image(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pgm", delete=False) as f1, \
             tempfile.NamedTemporaryFile("w", suffix=".pgm", delete=False) as f2:
            f1.write(pgm_ramp(18, 16))
            f2.write(pgm_ramp_rev(18, 16))
            p1, p2 = f1.name, f2.name
        self.assertGreater(hamming(dhash(p1), dhash(p2)), 5)
        os.unlink(p1); os.unlink(p2)


class TestAudit(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "train"))
        os.makedirs(os.path.join(self.root, "val"))

    def _img(self, rel, content):
        p = os.path.join(self.root, rel)
        with open(p, "w") as f:
            f.write(content)
        return p

    def test_exact_dup_and_leakage(self):
        img = pgm_ramp(18, 16)
        a = self._img("train/a.pgm", img)
        b = self._img("val/a_copy.pgm", img)
        findings = audit_directory(self.root)
        kinds = {f.kind for f in findings}
        self.assertIn("exact_dup", kinds)
        self.assertIn("leakage", kinds)

    def test_near_dup_not_flagged_when_far(self):
        self._img("train/a.pgm", pgm_ramp(18, 16))
        self._img("train/b.pgm", pgm_ramp_rev(18, 16))
        findings = audit_directory(self.root)
        self.assertNotIn("near_dup", {f.kind for f in findings})

    def test_missing_label(self):
        self._img("train/lonely.pgm", pgm_ramp(18, 16))
        findings = audit_directory(self.root)
        self.assertIn("missing_image", {f.kind for f in findings})

    def test_class_imbalance(self):
        for i in range(11):
            self._img(f"train/a{i}.pgm", pgm_ascii(9, 8, base=i, step=3))
        self._img("train/b0.pgm", pgm_ascii(9, 8, base=99, step=5))
        with open(os.path.join(self.root, "train", "labels.csv"), "w") as f:
            f.write("image,class\n")
            for i in range(11):
                f.write(f"a{i}.pgm,cat\n")
            f.write("b0.pgm,dog\n")
        findings = audit_directory(self.root)
        imb = [f for f in findings if f.kind == "class_imbalance"]
        self.assertEqual(len(imb), 1)
        self.assertIn("11", imb[0].detail)


if __name__ == "__main__":
    unittest.main()
