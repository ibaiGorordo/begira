from __future__ import annotations

from begira.server import create_app


def _skip(msg: str) -> None:  # pragma: no cover
    try:
        import pytest  # type: ignore

        pytest.skip(msg)
    except Exception:
        raise RuntimeError(msg)


def test_camera_api_accepts_intrinsics_and_size() -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f"TestClient not available ({e!r}); install test extras to run this test")
        return

    client = TestClient(create_app())
    create = client.post(
        "/api/elements/cameras",
        json={
            "name": "cam_api",
            "width": 1280,
            "height": 720,
            "intrinsicMatrix": [[600.0, 0.0, 640.0], [0.0, 610.0, 360.0], [0.0, 0.0, 1.0]],
            "fov": None,
        },
    )
    assert create.status_code == 200
    cam_id = create.json()["id"]

    meta = client.get(f"/api/elements/{cam_id}/meta")
    assert meta.status_code == 200
    data = meta.json()
    assert data["type"] == "camera"
    assert data["width"] == 1280
    assert data["height"] == 720
    assert data["intrinsicMatrix"] == [[600.0, 0.0, 640.0], [0.0, 610.0, 360.0], [0.0, 0.0, 1.0]]

