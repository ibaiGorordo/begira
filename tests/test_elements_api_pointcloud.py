from __future__ import annotations

import numpy as np


def _skip(msg: str) -> None:  # pragma: no cover
    try:
        import pytest  # type: ignore

        pytest.skip(msg)
    except Exception:
        raise RuntimeError(msg)


def test_elements_api_lists_and_serves_pointcloud_payload() -> None:
    from begira.server import create_app
    from begira.registry import REGISTRY

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f"TestClient not available ({e!r}); install test extras to run this test")
        return

    # Seed a tiny pointcloud.
    pos = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    col = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    pc = REGISTRY.upsert_pointcloud(name="pc", positions=pos, colors=col, element_id="pc1")

    client = TestClient(create_app())

    res = client.get("/api/elements")
    assert res.status_code == 200
    data = res.json()
    assert any(e["id"] == pc.id and e["type"] == "pointcloud" for e in data)

    meta = client.get(f"/api/elements/{pc.id}/meta").json()
    assert meta["type"] == "pointcloud"
    assert meta["pointCount"] == 2
    assert "payloads" in meta and "points" in meta["payloads"]

    payload_url = meta["payloads"]["points"]["url"]
    raw = client.get(payload_url).content

    # Interleaved XYZ(float32) + RGB(uint8) = 15 bytes/pt.
    assert len(raw) == 2 * 15
