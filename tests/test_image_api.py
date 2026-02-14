from __future__ import annotations

import uuid


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


def test_image_api_upload_and_payload() -> None:
    from begira.server import create_app

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f"TestClient not available ({e!r}); install test extras to run this test")
        return

    client = TestClient(create_app())

    req = client.post(
        "/api/elements/images/upload",
        json={
            "name": "img_api",
            "width": 1,
            "height": 1,
            "channels": 4,
            "mimeType": "image/png",
            "elementId": f"img_api_test_{uuid.uuid4().hex}",
        },
    )
    assert req.status_code == 200
    upload_url = req.json()["uploadUrl"]

    put = client.put(
        upload_url,
        params={
            "name": "img_api",
            "mimeType": "image/png",
            "width": "1",
            "height": "1",
            "channels": "4",
        },
        content=_PNG_1X1_RGBA,
        headers={"content-type": "application/octet-stream"},
    )
    assert put.status_code == 200
    image_id = put.json()["id"]

    elements = client.get("/api/elements")
    assert elements.status_code == 200
    listed = elements.json()
    assert any(e["id"] == image_id and e["type"] == "image" for e in listed)

    meta = client.get(f"/api/elements/{image_id}/meta")
    assert meta.status_code == 200
    m = meta.json()
    assert m["type"] == "image"
    assert m["width"] == 1
    assert m["height"] == 1
    assert m["channels"] == 4
    assert m["mimeType"] == "image/png"
    assert m["payloads"]["image"]["url"] == f"/api/elements/{image_id}/payloads/image"

    payload = client.get(m["payloads"]["image"]["url"])
    assert payload.status_code == 200
    assert payload.headers.get("content-type", "").startswith("image/png")
    assert payload.content == _PNG_1X1_RGBA
