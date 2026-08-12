"""EXIF / metadata read, edit, and strip using Pillow + piexif (offline)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

try:
    import piexif
except ImportError:  # pragma: no cover
    piexif = None  # type: ignore


def read_exif(source: str | Path | bytes | Image.Image) -> dict[str, Any]:
    """
    Read all available metadata from an image.

    Returns a flat dict of human-readable keys → values, plus a `_raw` summary.
    """
    img = _open(source)
    meta: dict[str, Any] = {
        "format": img.format,
        "mode": img.mode,
        "size": {"width": img.size[0], "height": img.size[1]},
    }

    # Basic info dict
    for k, v in (img.info or {}).items():
        if k in ("exif", "icc_profile", "xmp"):
            continue
        meta[f"info.{k}"] = _safe_value(v)

    # EXIF via getexif
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001
        exif = None

    if exif:
        for tag_id, value in exif.items():
            name = TAGS.get(tag_id, f"Tag_{tag_id}")
            if name == "GPSInfo" and isinstance(value, dict):
                for gk, gv in value.items():
                    gname = GPSTAGS.get(gk, f"GPS_{gk}")
                    meta[f"GPS.{gname}"] = _safe_value(gv)
            else:
                meta[f"EXIF.{name}"] = _safe_value(value)

    # piexif dump if available (more complete)
    if piexif is not None:
        try:
            raw = img.info.get("exif")
            if raw:
                ed = piexif.load(raw)
                for ifd_name in ("0th", "Exif", "GPS", "1st"):
                    ifd = ed.get(ifd_name) or {}
                    for tag, value in ifd.items():
                        try:
                            tname = piexif.TAGS[ifd_name][tag]["name"]
                        except Exception:  # noqa: BLE001
                            tname = str(tag)
                        key = f"piexif.{ifd_name}.{tname}"
                        if key not in meta:
                            meta[key] = _safe_value(value)
        except Exception:  # noqa: BLE001
            pass

    return meta


def strip_exif(source: str | Path | bytes | Image.Image) -> bytes:
    """Return image bytes with all EXIF / metadata stripped."""
    img = _open(source)
    fmt = (img.format or "PNG").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in ("JPEG", "PNG", "WEBP", "BMP", "TIFF"):
        fmt = "PNG"

    # Pixel-copy into a fresh image so info/exif are not carried over
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        pixels = img.convert("RGB")
    else:
        pixels = img.copy()

    # Fresh image from pixel data only — drops EXIF/info dict
    import numpy as np

    arr = np.array(pixels)
    clean = Image.fromarray(arr)

    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = 95
    clean.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def write_image_without_exif(
    source: str | Path | bytes | Image.Image,
    dest: str | Path,
) -> Path:
    """Strip metadata and write to dest path."""
    data = strip_exif(source)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def update_exif(
    source: str | Path | bytes | Image.Image,
    updates: dict[str, Any],
) -> bytes:
    """
    Best-effort field updates via piexif.

    `updates` keys should be piexif-style like "0th.ImageDescription" or
    simple "Artist", "Copyright", "ImageDescription", "Software".
    """
    img = _open(source)
    fmt = (img.format or "JPEG").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    if piexif is None or fmt not in ("JPEG", "TIFF"):
        # Fallback: strip all if we cannot edit
        return strip_exif(img)

    try:
        raw = img.info.get("exif", b"")
        exif_dict = piexif.load(raw) if raw else {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    except Exception:  # noqa: BLE001
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    simple_map = {
        "Artist": (piexif.ImageIFD.Artist, "0th"),
        "Copyright": (piexif.ImageIFD.Copyright, "0th"),
        "ImageDescription": (piexif.ImageIFD.ImageDescription, "0th"),
        "Software": (piexif.ImageIFD.Software, "0th"),
        "Make": (piexif.ImageIFD.Make, "0th"),
        "Model": (piexif.ImageIFD.Model, "0th"),
        "DateTime": (piexif.ImageIFD.DateTime, "0th"),
    }

    for key, value in updates.items():
        if value is None:
            continue
        if key in simple_map:
            tag, ifd = simple_map[key]
            encoded = value.encode("utf-8") if isinstance(value, str) else value
            exif_dict[ifd][tag] = encoded
        elif "." in key:
            ifd_name, tname = key.split(".", 1)
            if ifd_name in exif_dict and ifd_name in piexif.TAGS:
                tag_id = None
                for tid, info in piexif.TAGS[ifd_name].items():
                    if info["name"] == tname:
                        tag_id = tid
                        break
                if tag_id is not None:
                    encoded = value.encode("utf-8") if isinstance(value, str) else value
                    exif_dict[ifd_name][tag_id] = encoded

    exif_bytes = piexif.dump(exif_dict)
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L") and fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt, exif=exif_bytes, quality=95)
    return buf.getvalue()


def _open(source: str | Path | bytes | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


def _safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return value.hex()[:64]
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)
