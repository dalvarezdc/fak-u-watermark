"""C2PA / Content Credentials detection and best-effort strip.

MVP approach (no heavy native c2pa SDK required):
- Detect known markers in file bytes (c2pa, jumb, xmp gbd, etc.)
- Report status for UI
- Strip by re-encoding pixels without metadata containers (same as full strip)
  which removes many C2PA manifests embedded as XMP/APP11 for common formats.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

# Binary signatures often present in C2PA-bearing files
_MARKERS = [
    b"c2pa",
    b"C2PA",
    b"jumb",  # JUMBF box
    b"JUMB",
    b"c2ma",  # claim
    b"c2cl",
    b"c2cs",
    b"c2as",
    b"http://ns.adobe.com/xap/1.0/",
    b"http://ns.useplus.org/ldf/xmp/1.0/",
]


@dataclass
class C2PAReport:
    present: bool
    markers_found: list[str]
    format: str | None
    size: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "markers_found": self.markers_found,
            "format": self.format,
            "size": self.size,
            "notes": self.notes,
            "verdict": (
                "Possible C2PA / Content Credentials markers found"
                if self.present
                else "No C2PA markers detected"
            ),
        }


def detect_c2pa(source: str | Path | bytes) -> C2PAReport:
    raw, fmt = _read_bytes(source)
    found: list[str] = []
    for m in _MARKERS:
        if m in raw:
            try:
                label = m.decode("ascii", errors="replace")
            except Exception:  # noqa: BLE001
                label = repr(m)
            if label not in found:
                found.append(label)

    # Filter weak XMP-only hits: require c2pa/jumb OR multiple markers
    strong = [f for f in found if f.lower() in ("c2pa", "jumb", "c2ma", "c2cl", "c2cs", "c2as")]
    present = bool(strong) or (len(found) >= 2 and any("c2pa" in f.lower() for f in found))
    # If only adobe xap, not necessarily C2PA
    if not strong and found == ["http://ns.adobe.com/xap/1.0/"]:
        present = False
        notes = "XMP present but no strong C2PA/JUMBF markers."
    elif present:
        notes = (
            "Markers suggest Content Credentials / C2PA-related metadata. "
            "Full cryptographic verification needs official C2PA tooling."
        )
    else:
        notes = "No strong C2PA markers in file bytes."

    return C2PAReport(
        present=present,
        markers_found=found,
        format=fmt,
        size=len(raw),
        notes=notes,
    )


def strip_c2pa(source: str | Path | bytes) -> tuple[bytes, C2PAReport]:
    """
    Best-effort remove C2PA by re-encoding image pixels only.

    Returns (cleaned_bytes, post_strip_report).
    """
    from .exif import strip_exif

    before = detect_c2pa(source)
    cleaned = strip_exif(source)
    after = detect_c2pa(cleaned)
    after.notes = (
        f"Before: {before.verdict if hasattr(before, 'verdict') else before.present}. "
        f"After strip: {'still has markers' if after.present else 'no markers detected'}. "
        "Re-encode removes most embedded manifests for JPEG/PNG/WebP."
    )
    # use to_dict verdict
    after.notes = (
        f"Pre-strip present={before.present} markers={before.markers_found}. "
        f"Post-strip present={after.present} markers={after.markers_found}."
    )
    return cleaned, after


def _read_bytes(source: str | Path | bytes) -> tuple[bytes, str | None]:
    if isinstance(source, bytes):
        fmt = None
        try:
            fmt = Image.open(io.BytesIO(source)).format
        except Exception:  # noqa: BLE001
            pass
        return source, fmt
    path = Path(source)
    raw = path.read_bytes()
    fmt = None
    try:
        fmt = Image.open(io.BytesIO(raw)).format
    except Exception:  # noqa: BLE001
        pass
    return raw, fmt
