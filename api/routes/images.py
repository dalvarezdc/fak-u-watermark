"""Image EXIF and inpainting endpoints."""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.deps import get_history_store

router = APIRouter(prefix="/images", tags=["images"])


class ExifUpdateRequest(BaseModel):
    image_b64: str
    updates: dict[str, Any] = Field(default_factory=dict)


class InpaintRequest(BaseModel):
    image_b64: str
    mask_b64: str | None = None
    # Rectangle alternative to full mask
    x1: int | None = None
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None
    instruction: str | None = None
    method: str = "telea"
    radius: int = 5


def _decode_b64(data: str) -> bytes:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


@router.post("/exif")
async def read_image_exif(file: UploadFile = File(...)) -> dict[str, Any]:
    from image_tools.exif import read_exif

    raw = await file.read()
    try:
        meta = read_exif(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_history_store()
    store.add(
        kind="image",
        title=f"EXIF: {file.filename or 'upload'}",
        payload={"type": "exif", "filename": file.filename, "meta_keys": list(meta.keys())},
    )
    return {"filename": file.filename, "metadata": meta}


@router.post("/exif/b64")
def read_exif_b64(body: dict[str, str]) -> dict[str, Any]:
    from image_tools.exif import read_exif

    try:
        raw = _decode_b64(body["image_b64"])
        meta = read_exif(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"metadata": meta}


@router.post("/strip-exif")
async def strip_image_exif(file: UploadFile = File(...)) -> Response:
    from image_tools.exif import strip_exif

    raw = await file.read()
    try:
        cleaned = strip_exif(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_history_store()
    store.add(
        kind="image",
        title=f"Strip EXIF: {file.filename or 'upload'}",
        payload={"type": "strip_exif", "filename": file.filename},
    )
    media = "image/png"
    fname = (file.filename or "cleaned").rsplit(".", 1)[0] + "_stripped.png"
    return Response(
        content=cleaned,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.post("/strip-exif/b64")
def strip_exif_b64(body: dict[str, str]) -> dict[str, Any]:
    from image_tools.exif import strip_exif

    try:
        raw = _decode_b64(body["image_b64"])
        cleaned = strip_exif(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "image_b64": base64.b64encode(cleaned).decode("ascii"),
        "format": "PNG",
    }


@router.post("/update-exif")
def update_image_exif(body: ExifUpdateRequest) -> dict[str, Any]:
    from image_tools.exif import update_exif

    try:
        raw = _decode_b64(body.image_b64)
        updated = update_exif(raw, body.updates)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "image_b64": base64.b64encode(updated).decode("ascii"),
        "format": "JPEG",
    }


@router.post("/inpaint")
def inpaint_image(body: InpaintRequest) -> dict[str, Any]:
    from image_tools.inpaint import inpaint_region, rectangle_mask
    from PIL import Image
    import io

    try:
        raw = _decode_b64(body.image_b64)
        if body.mask_b64:
            mask = _decode_b64(body.mask_b64)
        elif None not in (body.x1, body.y1, body.x2, body.y2):
            img = Image.open(io.BytesIO(raw))
            w, h = img.size
            mask = rectangle_mask(w, h, body.x1 or 0, body.y1 or 0, body.x2 or 0, body.y2 or 0)
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide mask_b64 or rectangle coordinates (x1,y1,x2,y2).",
            )

        result = inpaint_region(
            raw,
            mask,
            instruction=body.instruction,
            method=body.method,
            radius=body.radius,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not result.success or not result.image_bytes:
        raise HTTPException(status_code=500, detail=result.error or "Inpaint failed")

    store = get_history_store()
    store.add(
        kind="image",
        title="Inpaint region",
        payload={
            "type": "inpaint",
            "method": result.method,
            "instruction": body.instruction,
        },
    )
    return {
        "image_b64": base64.b64encode(result.image_bytes).decode("ascii"),
        "format": result.format,
        "method": result.method,
    }


@router.post("/inpaint/upload")
async def inpaint_upload(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    x1: int | None = Form(None),
    y1: int | None = Form(None),
    x2: int | None = Form(None),
    y2: int | None = Form(None),
    instruction: str | None = Form(None),
    method: str = Form("telea"),
    radius: int = Form(5),
) -> Response:
    from image_tools.inpaint import inpaint_region, rectangle_mask
    from PIL import Image
    import io

    raw = await image.read()
    if mask is not None:
        mask_data: Any = await mask.read()
    elif None not in (x1, y1, x2, y2):
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        mask_data = rectangle_mask(w, h, x1 or 0, y1 or 0, x2 or 0, y2 or 0)
    else:
        raise HTTPException(status_code=400, detail="Provide mask file or rectangle coords.")

    result = inpaint_region(
        raw, mask_data, instruction=instruction, method=method, radius=radius
    )
    if not result.success or not result.image_bytes:
        raise HTTPException(status_code=500, detail=result.error or "Inpaint failed")

    return Response(
        content=result.image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="inpainted.png"'},
    )


@router.get("/history")
def image_history(limit: int = 50) -> dict[str, Any]:
    store = get_history_store()
    entries = store.list(kind="image", limit=limit)
    return {"entries": [e.to_dict() for e in entries]}
