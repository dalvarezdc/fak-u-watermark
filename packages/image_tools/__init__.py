"""Image metadata (EXIF) and visual watermark removal helpers."""

from .exif import read_exif, strip_exif, update_exif, write_image_without_exif
from .inpaint import inpaint_region

__all__ = [
    "read_exif",
    "strip_exif",
    "update_exif",
    "write_image_without_exif",
    "inpaint_region",
]
