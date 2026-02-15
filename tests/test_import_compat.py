from __future__ import annotations


def test_package_import_paths_work() -> None:
    from begira.api import create_api_app
    from begira.api.routes import mount_elements_api
    from begira.core.registry import REGISTRY, InMemoryRegistry
    from begira.io.image import encode_image_payload
    from begira.io.ply import load_ply
    from begira.runtime.server import BegiraServer, run
    from begira.sdk.client import BegiraClient
    from begira.sdk.handles import CameraHandle

    assert create_api_app is not None
    assert mount_elements_api is not None
    assert REGISTRY is not None
    assert InMemoryRegistry is not None
    assert BegiraClient is not None
    assert CameraHandle is not None
    assert BegiraServer is not None
    assert run is not None
    assert load_ply is not None
    assert encode_image_payload is not None
