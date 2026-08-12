"""Image metadata (EXIF) and visual watermark removal helpers."""

from .c2pa_tools import detect_c2pa, strip_c2pa
from .exif import read_exif, strip_exif, update_exif, write_image_without_exif
from .inpaint import extract_mask_from_editor, inpaint_region, rectangle_mask

__all__ = [
    "read_exif",
    "strip_exif",
    "update_exif",
    "write_image_without_exif",
    "inpaint_region",
    "rectangle_mask",
    "extract_mask_from_editor",
    "detect_c2pa",
    "strip_c2pa",
]
