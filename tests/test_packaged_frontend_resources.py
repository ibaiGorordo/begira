from __future__ import annotations


def _skip(msg: str) -> None:  # pragma: no cover
    try:
        import pytest  # type: ignore

        pytest.skip(msg)
    except Exception:
        raise RuntimeError(msg)


def test_packaged_frontend_dist_exists_when_installed() -> None:
    """This is mainly to be run from an installed wheel.

    In editable/dev checkouts you might not have run the packaging step that copies
    `frontend/dist` into `begira/_frontend/dist`.
    """

    try:
        from importlib import resources as importlib_resources
    except Exception as e:  # pragma: no cover
        _skip(f"importlib.resources not available: {e!r}")
        return

    try:
        dist = importlib_resources.files("begira._frontend").joinpath("dist")
    except ModuleNotFoundError as e:  # pragma: no cover
        _skip(f"begira._frontend package missing: {e!r}")
        return

    # This asserts the wheel included the Vite build output.
    assert dist.joinpath("index.html").is_file()
