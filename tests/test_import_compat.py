from __future__ import annotations


def test_legacy_import_paths_still_work() -> None:
    import begira.animation as legacy_animation
    import begira.api_time as legacy_api_time
    import begira.client as legacy_client
    import begira.conventions as legacy_conventions
    import begira.element_projection as legacy_projection
    import begira.elements as legacy_elements
    import begira.elements_api as legacy_elements_api
    import begira.handles as legacy_handles
    import begira.image_logging as legacy_image_logging
    import begira.ply as legacy_ply
    import begira.registry as legacy_registry
    import begira.runner as legacy_runner
    import begira.server as legacy_server
    import begira.timeline as legacy_timeline
    import begira.viewer_settings as legacy_viewer_settings
    import begira.web as legacy_web

    assert legacy_client.BegiraClient is not None
    assert legacy_runner.run is not None
    assert legacy_server.create_app is not None
    assert legacy_elements_api.mount_elements_api is not None
    assert legacy_registry.REGISTRY is not None
    assert legacy_conventions.CoordinateConvention is not None
    assert legacy_timeline.TemporalChannel is not None
    assert legacy_animation.CameraAnimationTrack is not None
    assert legacy_handles.CameraHandle is not None
    assert legacy_ply.load_ply is not None
    assert legacy_image_logging.encode_image_payload is not None
    assert legacy_projection.element_to_meta_item is not None
    assert legacy_elements.ElementBase is not None
    assert legacy_api_time.parse_sample_query is not None
    assert legacy_viewer_settings.VIEWER_SETTINGS is not None
    assert legacy_web.mount_frontend is not None


def test_new_package_paths_work() -> None:
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
