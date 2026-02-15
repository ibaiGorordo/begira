from __future__ import annotations


def _skip(msg: str) -> None:  # pragma: no cover
    # Local import to avoid requiring pytest in environments where only runtime deps are installed.
    try:
        import pytest  # type: ignore

        pytest.skip(msg)
    except Exception:
        raise RuntimeError(msg)


def test_root_serves_packaged_frontend_index_html() -> None:
    """The root URL should serve the packaged Vite index.html."""

    from begira.runtime.app import create_app

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        _skip(f"TestClient not available ({e!r}); install test extras to run this test")
        return

    client = TestClient(create_app())
    res = client.get("/")

    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")

    # Vite's index.html should include a module script tag.
    body = res.text.lower()
    assert "<script" in body
    assert "type=\"module\"" in body or "type='module'" in body
