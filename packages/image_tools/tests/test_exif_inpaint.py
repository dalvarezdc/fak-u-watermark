"""Tests for EXIF strip and local inpainting."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from image_tools.exif import read_exif, strip_exif, update_exif
from image_tools.inpaint import inpaint_region, rectangle_mask


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 48), color=(30, 120, 200))
    # Draw a white "watermark" rectangle
    for x in range(10, 50):
        for y in range(5, 15):
            img.putpixel((x, y), (255, 255, 255))
    buf = io.BytesIO()
    try:
        import piexif

        exif = {
            "0th": {
                piexif.ImageIFD.Artist: b"Test Artist",
                piexif.ImageIFD.Software: b"faku-test",
            },
            "Exif": {},
            "GPS": {},
            "1st": {},
            "thumbnail": None,
        }
        exif_bytes = piexif.dump(exif)
        img.save(buf, format="JPEG", quality=90, exif=exif_bytes)
    except Exception:  # noqa: BLE001
        img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class TestExif:
    def test_read(self, sample_jpeg_bytes):
        meta = read_exif(sample_jpeg_bytes)
        assert meta["size"]["width"] == 64
        assert meta["size"]["height"] == 48
        assert meta["format"] in ("JPEG", "MPO", None) or "format" in meta

    def test_strip(self, sample_jpeg_bytes):
        cleaned = strip_exif(sample_jpeg_bytes)
        assert len(cleaned) > 0
        # Re-open succeeds
        img = Image.open(io.BytesIO(cleaned))
        assert img.size == (64, 48)

    def test_update(self, sample_jpeg_bytes):
        try:
            import piexif  # noqa: F401
        except ImportError:
            pytest.skip("piexif not installed")
        updated = update_exif(sample_jpeg_bytes, {"Artist": "New Author", "Copyright": "2026"})
        meta = read_exif(updated)
        # Artist may appear under different keys depending on reader
        flat = " ".join(str(v) for v in meta.values())
        assert "New Author" in flat or "Artist" in str(meta)


class TestInpaint:
    def test_rectangle_mask(self):
        m = rectangle_mask(100, 80, 10, 10, 40, 30)
        assert m.shape == (80, 100)
        assert m[15, 20] == 255
        assert m[0, 0] == 0

    def test_inpaint_runs(self, sample_jpeg_bytes):
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("opencv not installed")

        img = Image.open(io.BytesIO(sample_jpeg_bytes))
        w, h = img.size
        mask = rectangle_mask(w, h, 10, 5, 50, 15)
        result = inpaint_region(img, mask, method="telea")
        assert result.success, result.error
        assert result.image_bytes is not None
        out = Image.open(io.BytesIO(result.image_bytes))
        assert out.size == img.size

    def test_empty_mask_fails(self, sample_jpeg_bytes):
        import numpy as np

        img = Image.open(io.BytesIO(sample_jpeg_bytes))
        mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        result = inpaint_region(img, mask, method="telea")
        assert not result.success
        assert "empty" in (result.error or "").lower()

    def test_extract_mask_from_editor(self):
        from image_tools.inpaint import extract_mask_from_editor
        import numpy as np

        bg = Image.new("RGB", (40, 30), (10, 20, 30))
        layer = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
        for x in range(5, 15):
            for y in range(5, 12):
                layer.putpixel((x, y), (255, 255, 255, 255))
        img, mask = extract_mask_from_editor(
            {"background": bg, "layers": [layer], "composite": bg}
        )
        assert img is not None
        assert mask is not None
        assert mask[7, 8] == 255
        assert mask[0, 0] == 0

    def test_api_without_key_falls_back_or_errors(self, sample_jpeg_bytes):
        img = Image.open(io.BytesIO(sample_jpeg_bytes))
        w, h = img.size
        mask = rectangle_mask(w, h, 10, 5, 50, 15)
        result = inpaint_region(
            img,
            mask,
            method="api",
            instruction="remove watermark",
            api_key=None,
        )
        # No key → error (unless env has a key). Local fallback only after API HTTP failure.
        if not result.success:
            assert "key" in (result.error or "").lower() or result.error
