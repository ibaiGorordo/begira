from __future__ import annotations

from io import BytesIO

import numpy as np


_MIME_TO_FORMAT: dict[str, str] = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
    "image/webp": "WEBP",
}

_MIME_TO_CV2_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}


def _normalize_mime_type(mime_type: str | None) -> str:
    mime = str(mime_type or "image/png").strip().lower()
    if mime not in _MIME_TO_FORMAT:
        raise ValueError(f"Unsupported mime_type {mime!r}. Supported: {sorted(_MIME_TO_FORMAT.keys())}")
    return mime


def _coerce_u8_image(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim not in (2, 3):
        raise ValueError(f"image array must have shape (H,W) or (H,W,C), got {a.shape}")
    if a.ndim == 3 and a.shape[2] not in (1, 3, 4):
        raise ValueError(f"image array channel count must be 1, 3, or 4; got {a.shape[2]}")
    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, 0.0, 1.0) * 255.0
    else:
        a = np.clip(a, 0, 255)
    return np.ascontiguousarray(a, dtype=np.uint8)


def _encode_numpy_with_cv2(arr_u8: np.ndarray, mime_type: str, *, color_order: str) -> bytes:
    import cv2  # type: ignore

    ext = _MIME_TO_CV2_EXT[mime_type]
    img = arr_u8
    # OpenCV expects BGR/BGRA data for color images.
    if arr_u8.ndim == 3 and arr_u8.shape[2] == 3 and color_order == "rgb":
        img = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)
    elif arr_u8.ndim == 3 and arr_u8.shape[2] == 4 and color_order == "rgb":
        img = cv2.cvtColor(arr_u8, cv2.COLOR_RGBA2BGRA)

    ok, enc = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"cv2.imencode failed for mime_type={mime_type!r}")
    return bytes(enc.tobytes())


def _encode_numpy_with_pillow(arr_u8: np.ndarray, mime_type: str, *, color_order: str) -> bytes:
    from PIL import Image  # type: ignore

    out = arr_u8
    mode: str
    if arr_u8.ndim == 2:
        mode = "L"
    else:
        ch = int(arr_u8.shape[2])
        if ch == 1:
            mode = "L"
            out = arr_u8[:, :, 0]
        elif ch == 3:
            mode = "RGB"
            if color_order == "bgr":
                out = arr_u8[:, :, ::-1]
        elif ch == 4:
            mode = "RGBA"
            if color_order == "bgr":
                out = arr_u8[:, :, [2, 1, 0, 3]]
        else:
            raise ValueError(f"Unsupported channel count: {ch}")

    img = Image.fromarray(out, mode=mode)
    buf = BytesIO()
    img.save(buf, format=_MIME_TO_FORMAT[mime_type])
    return bytes(buf.getvalue())


def encode_image_payload(
    image: object,
    *,
    mime_type: str | None = "image/png",
    color_order: str = "bgr",
    width: int | None = None,
    height: int | None = None,
    channels: int | None = None,
) -> tuple[bytes, str, int, int, int]:
    """Encode an image payload suitable for logging.

    Supports:
    - numpy arrays (OpenCV-style ndarray or RGB/RGBA arrays)
    - PIL images (objects exposing `.save()` and `.size`)
    """
    mime = _normalize_mime_type(mime_type)
    order = str(color_order).lower()
    if order not in {"bgr", "rgb"}:
        raise ValueError("color_order must be 'bgr' or 'rgb'")

    if isinstance(image, (bytes, bytearray, memoryview)):
        if width is None or height is None or channels is None:
            raise ValueError("width, height, and channels are required when logging pre-encoded image bytes")
        width_i = int(width)
        height_i = int(height)
        channels_i = int(channels)
        if width_i <= 0 or height_i <= 0 or channels_i <= 0:
            raise ValueError("width, height, and channels must be positive integers")
        return bytes(image), mime, width_i, height_i, channels_i

    if isinstance(image, np.ndarray):
        arr_u8 = _coerce_u8_image(image)
        height = int(arr_u8.shape[0])
        width = int(arr_u8.shape[1])
        channels = int(arr_u8.shape[2]) if arr_u8.ndim == 3 else 1
        try:
            data = _encode_numpy_with_cv2(arr_u8, mime, color_order=order)
        except Exception:
            try:
                data = _encode_numpy_with_pillow(arr_u8, mime, color_order=order)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to encode numpy image. Install OpenCV (`opencv-python`) or Pillow (`Pillow`)."
                ) from exc
        return data, mime, width, height, channels

    if hasattr(image, "save") and hasattr(image, "size"):
        size = getattr(image, "size")
        if not isinstance(size, tuple) or len(size) != 2:
            raise ValueError("PIL-like image has invalid size")
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError("PIL-like image has invalid dimensions")
        mode = str(getattr(image, "mode", "RGB"))
        channels = {
            "1": 1,
            "L": 1,
            "LA": 2,
            "P": 1,
            "RGB": 3,
            "RGBA": 4,
            "CMYK": 4,
        }.get(mode, max(1, len(mode)))
        buf = BytesIO()
        image.save(buf, format=_MIME_TO_FORMAT[mime])
        return bytes(buf.getvalue()), mime, width, height, int(channels)

    raise TypeError(
        "Unsupported image type. Expected numpy.ndarray (OpenCV-style) or PIL Image."
    )
