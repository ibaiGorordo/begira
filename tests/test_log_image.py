from __future__ import annotations

import numpy as np

from begira.registry import REGISTRY
from begira.runner import BegiraServer


def _skip(msg: str) -> None:  # pragma: no cover
    try:
        import pytest  # type: ignore

        pytest.skip(msg)
    except Exception:
        raise RuntimeError(msg)


_PNG_1X1_RGBA = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0cIDATx\x9cc```\xf8\x0f\x00\x01\x04\x01\x00\x18\xdd\x8d\xb5"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_log_image_from_preencoded_bytes() -> None:
    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")

    handle = server.log_image(
        "img0",
        _PNG_1X1_RGBA,
        mime_type="image/png",
        width=1,
        height=1,
        channels=4,
    )
    stored = REGISTRY.get_element(handle)
    assert stored is not None
    assert stored.type == "image"

    assert handle.width == 1
    assert handle.height == 1
    assert handle.channels == 4
    assert handle.mime_type == "image/png"

    meta = handle.meta
    assert meta["type"] == "image"
    assert meta["width"] == 1
    assert meta["height"] == 1
    assert "payloads" in meta and "image" in meta["payloads"]


def test_log_image_from_numpy_array_when_encoder_available() -> None:
    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception:
        try:
            from PIL import Image  # type: ignore  # noqa: F401
        except Exception:
            _skip("Neither OpenCV nor Pillow is available; install one to run numpy image logging test")
            return

    server = BegiraServer(host="127.0.0.1", port=0, url="http://127.0.0.1:0/")
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    handle = server.log_image("img_numpy", img)
    assert handle.width == 8
    assert handle.height == 6
    assert handle.channels == 3
