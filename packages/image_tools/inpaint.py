"""Region inpainting for visible watermark removal.

MVP: local OpenCV Telea/NS inpainting (offline).
Optional: external API hook via env for higher quality later.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class InpaintResult:
    success: bool
    image_bytes: bytes | None
    format: str = "PNG"
    method: str = "opencv_telea"
    error: str | None = None
    instruction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "format": self.format,
            "method": self.method,
            "error": self.error,
            "instruction": self.instruction,
            "has_image": self.image_bytes is not None,
        }


def inpaint_region(
    image: str | Path | bytes | Image.Image,
    mask: str | Path | bytes | Image.Image | np.ndarray,
    *,
    instruction: str | None = None,
    method: str = "telea",
    radius: int = 5,
) -> InpaintResult:
    """
    Inpaint pixels covered by mask (non-zero = remove).

    `instruction` is accepted for API compatibility; local OpenCV ignores it.
    Mask should be same size as image; white/non-zero marks the watermark region.
    """
    try:
        import cv2
    except ImportError as exc:
        return InpaintResult(
            success=False,
            image_bytes=None,
            error=f"opencv-python required for local inpainting: {exc}",
            instruction=instruction,
        )

    try:
        img = _to_bgr(image)
        m = _to_mask(mask, img.shape[:2])
        flag = cv2.INPAINT_TELEA if method != "ns" else cv2.INPAINT_NS
        result = cv2.inpaint(img, m, radius, flag)
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return InpaintResult(
            success=True,
            image_bytes=buf.getvalue(),
            format="PNG",
            method=f"opencv_{method}",
            instruction=instruction,
        )
    except Exception as exc:  # noqa: BLE001
        return InpaintResult(
            success=False,
            image_bytes=None,
            error=str(exc),
            instruction=instruction,
        )


def rectangle_mask(
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    """Create a binary mask (uint8) for a rectangle region."""
    mask = np.zeros((height, width), dtype=np.uint8)
    xa, xb = sorted((max(0, x1), min(width, x2)))
    ya, yb = sorted((max(0, y1), min(height, y2)))
    mask[ya:yb, xa:xb] = 255
    return mask


def _to_bgr(source: str | Path | bytes | Image.Image) -> np.ndarray:
    import cv2

    if isinstance(source, Image.Image):
        arr = np.array(source.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img
    path = str(source)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback via PIL
        pil = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _to_mask(
    mask: str | Path | bytes | Image.Image | np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Return single-channel uint8 mask matching (height, width)."""
    import cv2

    h, w = size
    if isinstance(mask, np.ndarray):
        m = mask
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if m.shape[2] == 3 else m[:, :, 0]
    elif isinstance(mask, Image.Image):
        m = np.array(mask.convert("L"))
    elif isinstance(mask, bytes):
        arr = np.frombuffer(mask, dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError("Could not decode mask bytes")
    else:
        m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.array(Image.open(mask).convert("L"))

    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    # Any non-zero → 255
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    return m.astype(np.uint8)
