"""Region inpainting for visible watermark removal.

- Local: OpenCV Telea/NS (offline, default)
- API: OpenAI-compatible images/edits when key + instruction provided
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

InpaintMethod = Literal["telea", "ns", "api", "auto"]


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
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> InpaintResult:
    """
    Inpaint pixels covered by mask (non-zero = remove).

    method:
      - telea / ns: local OpenCV
      - api: OpenAI-compatible images/edits (needs key + prompt)
      - auto: try API if key+instruction present, else local Telea
    """
    method = (method or "telea").lower().strip()
    if method == "auto":
        key = api_key or os.environ.get("FAKU_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if key and (instruction or "").strip():
            method = "api"
        else:
            method = "telea"

    if method == "api":
        return inpaint_via_api(
            image,
            mask,
            instruction=instruction or "Remove the watermark or logo; fill naturally.",
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    return _inpaint_opencv(image, mask, method=method, radius=radius, instruction=instruction)


def _inpaint_opencv(
    image: str | Path | bytes | Image.Image,
    mask: str | Path | bytes | Image.Image | np.ndarray,
    *,
    method: str = "telea",
    radius: int = 5,
    instruction: str | None = None,
) -> InpaintResult:
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
        if int(m.sum()) == 0:
            return InpaintResult(
                success=False,
                image_bytes=None,
                error="Mask is empty — paint over the watermark first.",
                instruction=instruction,
            )
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


def inpaint_via_api(
    image: str | Path | bytes | Image.Image,
    mask: str | Path | bytes | Image.Image | np.ndarray,
    *,
    instruction: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 180.0,
) -> InpaintResult:
    """
    Call OpenAI-compatible POST /images/edits with image + alpha mask + prompt.

    Transparent (or low-alpha) mask pixels mark the region to edit.
    Our internal mask is white=edit; we convert to RGBA alpha=0 for edit region.
    """
    import httpx

    key = (
        api_key
        or os.environ.get("FAKU_INPAINT_API_KEY")
        or os.environ.get("FAKU_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not key:
        return InpaintResult(
            success=False,
            image_bytes=None,
            error="No API key for inpainting. Set FAKU_API_KEY / OPENAI_API_KEY.",
            instruction=instruction,
            method="api",
        )

    base = (
        base_url
        or os.environ.get("FAKU_INPAINT_BASE_URL")
        or os.environ.get("FAKU_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    # Some providers use a dedicated edits model
    model_name = (
        model
        or os.environ.get("FAKU_INPAINT_MODEL")
        or "dall-e-2"
    )

    try:
        img_png = _to_png_rgba_bytes(image)
        mask_png = _mask_to_openai_png(image, mask)
    except Exception as exc:  # noqa: BLE001
        return InpaintResult(
            success=False,
            image_bytes=None,
            error=f"Failed to prepare image/mask: {exc}",
            instruction=instruction,
            method="api",
        )

    url = f"{base}/images/edits"
    headers = {"Authorization": f"Bearer {key}"}
    files = {
        "image": ("image.png", img_png, "image/png"),
        "mask": ("mask.png", mask_png, "image/png"),
    }
    data: dict[str, Any] = {
        "prompt": instruction,
        "n": "1",
        "response_format": "b64_json",
    }
    # dall-e-2 requires square sizes; omit size to let provider default when possible
    if model_name:
        data["model"] = model_name

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, files=files, data=data)
            if resp.status_code >= 400:
                # Fall back to local if API fails
                detail = resp.text[:500]
                local = _inpaint_opencv(image, mask, method="telea", instruction=instruction)
                if local.success:
                    local.method = f"opencv_telea_fallback"
                    local.error = f"API failed ({resp.status_code}): {detail}; used local fallback."
                    # Still success with local
                    local.error = None
                    local.method = "opencv_telea (api fallback)"
                    return local
                return InpaintResult(
                    success=False,
                    image_bytes=None,
                    error=f"Inpaint API {resp.status_code}: {detail}",
                    instruction=instruction,
                    method="api",
                )
            payload = resp.json()
            items = payload.get("data") or []
            if not items:
                return InpaintResult(
                    success=False,
                    image_bytes=None,
                    error="API returned no image data.",
                    instruction=instruction,
                    method="api",
                )
            item = items[0]
            if "b64_json" in item:
                import base64

                raw = base64.b64decode(item["b64_json"])
            elif "url" in item:
                r2 = httpx.get(item["url"], timeout=timeout)
                r2.raise_for_status()
                raw = r2.content
            else:
                return InpaintResult(
                    success=False,
                    image_bytes=None,
                    error="Unexpected API response format.",
                    instruction=instruction,
                    method="api",
                )
            # Normalize to PNG
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            out = io.BytesIO()
            pil.save(out, format="PNG")
            return InpaintResult(
                success=True,
                image_bytes=out.getvalue(),
                format="PNG",
                method=f"api:{model_name}",
                instruction=instruction,
            )
    except Exception as exc:  # noqa: BLE001
        local = _inpaint_opencv(image, mask, method="telea", instruction=instruction)
        if local.success:
            local.method = "opencv_telea (api error fallback)"
            return local
        return InpaintResult(
            success=False,
            image_bytes=None,
            error=str(exc),
            instruction=instruction,
            method="api",
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


def extract_mask_from_editor(
    editor_value: Any,
) -> tuple[Image.Image | None, np.ndarray | None]:
    """
    Parse Gradio ImageMask / ImageEditor output into (background image, binary mask).

    Mask: uint8 HxW, 255 = region to remove.
    """
    if editor_value is None:
        return None, None

    if isinstance(editor_value, Image.Image):
        return editor_value.convert("RGB"), None

    if isinstance(editor_value, np.ndarray):
        return Image.fromarray(editor_value).convert("RGB"), None

    if not isinstance(editor_value, dict):
        return None, None

    # Common Gradio keys
    bg = editor_value.get("background")
    composite = editor_value.get("composite")
    layers = editor_value.get("layers") or []
    mask_raw = editor_value.get("mask")

    base_img: Image.Image | None = None
    for candidate in (bg, composite, editor_value.get("image")):
        if candidate is None:
            continue
        base_img = _as_pil(candidate)
        if base_img is not None:
            break

    mask_arr: np.ndarray | None = None

    if mask_raw is not None:
        mimg = _as_pil(mask_raw)
        if mimg is not None:
            mask_arr = _pil_layer_to_mask(mimg)

    if mask_arr is None and layers:
        # Union of all non-transparent paint strokes
        h = w = None
        if base_img is not None:
            w, h = base_img.size
        acc = None
        for layer in layers:
            limg = _as_pil(layer)
            if limg is None:
                continue
            if h is None:
                w, h = limg.size
            part = _pil_layer_to_mask(limg)
            acc = part if acc is None else np.maximum(acc, part)
        mask_arr = acc

    # If composite differs from background, use difference as mask
    if mask_arr is None and bg is not None and composite is not None:
        b = np.array(_as_pil(bg).convert("RGB"))
        c = np.array(_as_pil(composite).convert("RGB"))
        if b.shape == c.shape:
            diff = np.abs(b.astype(int) - c.astype(int)).sum(axis=2)
            mask_arr = (diff > 15).astype(np.uint8) * 255

    if base_img is None and composite is not None:
        base_img = _as_pil(composite)

    if base_img is not None and mask_arr is not None:
        if mask_arr.shape[:2] != (base_img.size[1], base_img.size[0]):
            mask_arr = np.array(
                Image.fromarray(mask_arr).resize(base_img.size, Image.Resampling.NEAREST)
            )

    return base_img, mask_arr


def _as_pil(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return Image.fromarray(value)
        if value.shape[-1] == 4:
            return Image.fromarray(value, "RGBA")
        return Image.fromarray(value).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value)
    if isinstance(value, dict) and "path" in value:
        return Image.open(value["path"])
    return None


def _pil_layer_to_mask(layer: Image.Image) -> np.ndarray:
    """Non-transparent or bright pixels become 255."""
    if layer.mode in ("RGBA", "LA"):
        alpha = np.array(layer.getchannel("A"))
        return (alpha > 10).astype(np.uint8) * 255
    arr = np.array(layer.convert("L"))
    # Painted strokes often non-black
    return (arr > 10).astype(np.uint8) * 255


def _to_png_rgba_bytes(source: str | Path | bytes | Image.Image) -> bytes:
    img = _open_pil(source).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_to_openai_png(
    image: str | Path | bytes | Image.Image,
    mask: str | Path | bytes | Image.Image | np.ndarray,
) -> bytes:
    """
    OpenAI edits: transparent pixels = area to edit.
    Our mask: 255 = edit → alpha 0; 0 = keep → alpha 255.
    """
    base = _open_pil(image).convert("RGBA")
    w, h = base.size
    m = _to_mask(mask, (h, w))
    rgba = np.array(base)
    # Where mask is set, make fully transparent
    edit = m > 0
    rgba[edit, 3] = 0
    rgba[~edit, 3] = 255
    out = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _open_pil(source: str | Path | bytes | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


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
        if mask.mode in ("RGBA", "LA"):
            m = _pil_layer_to_mask(mask)
        else:
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
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    return m.astype(np.uint8)
