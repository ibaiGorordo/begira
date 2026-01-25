from __future__ import annotations
import numpy as np
import httpx
from begira.client import BegiraClient
from begira.runner import run

def test_log_gaussians_api() -> None:
    # Start a server in a thread
    server = run(port=0, open_browser=False)
    client = BegiraClient(server.url)
    
    n = 100
    positions = np.random.rand(n, 3).astype(np.float32)
    sh0 = np.random.rand(n, 3).astype(np.float32)
    opacity = np.random.rand(n, 1).astype(np.float32)
    scales = np.random.rand(n, 3).astype(np.float32)
    rotations = np.random.rand(n, 4).astype(np.float32)
    
    eid = client.log_gaussians(
        "test_gaussians",
        positions,
        sh0=sh0,
        opacity=opacity,
        scales=scales,
        rotations=rotations
    )
    
    assert eid is not None
    
    # Check elements list
    res = httpx.get(f"{server.url.rstrip('/')}/api/elements")
    assert res.status_code == 200
    elements = res.json()
    assert any(e["id"] == eid and e["type"] == "gaussians" for e in elements)
    
    # Check metadata
    res = httpx.get(f"{server.url.rstrip('/')}/api/elements/{eid}/meta")
    assert res.status_code == 200
    meta = res.json()
    assert meta["type"] == "gaussians"
    assert meta["count"] == n
    assert meta["pointSize"] == 1.0
    assert "gaussians" in meta["payloads"]
    
    # Check payload
    payload_url = meta["payloads"]["gaussians"]["url"]
    res = httpx.get(f"{server.url.rstrip('/')}{payload_url}")
    assert res.status_code == 200
    payload = res.content
    assert len(payload) == n * 14 * 4
    
    data = np.frombuffer(payload, dtype="<f4").reshape((n, 14))
    np.testing.assert_allclose(data[:, 0:3], positions)
    np.testing.assert_allclose(data[:, 3:6], sh0)
    np.testing.assert_allclose(data[:, 6:7], opacity)
    np.testing.assert_allclose(data[:, 7:10], scales)
    np.testing.assert_allclose(data[:, 10:14], rotations)
