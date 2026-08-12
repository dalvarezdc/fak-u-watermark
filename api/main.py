"""FastAPI entrypoint for fak-u-watermark."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import _ROOT, _PACKAGES  # noqa: F401 — side-effect path setup
from api.routes import images, text

app = FastAPI(
    title="fak-u-watermark",
    description=(
        "Detect, highlight, and neutralize AI text watermarks. "
        "Inspect/strip image EXIF and remove visual watermarks from regions. "
        "Strip the mark. Keep the meaning."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(text.router, prefix="/api")
app.include_router(images.router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "fak-u-watermark",
        "tagline": "Strip the mark. Keep the meaning.",
        "docs": "/docs",
        "endpoints": {
            "analyze": "POST /api/text/analyze",
            "neutralize": "POST /api/text/neutralize",
            "exif": "POST /api/images/exif",
            "strip_exif": "POST /api/images/strip-exif",
            "inpaint": "POST /api/images/inpaint",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
