from __future__ import annotations


def test_run_auto_attaches_to_existing_server() -> None:
    """If a server is reachable at host/port, begira.run() should attach by default.

    This keeps examples simple: users can write `client = begira.run(...)` and it
    will connect if possible, otherwise start a new server.
    """

    import begira

    # Start a server first on an explicit port.
    server = begira.run(host="127.0.0.1", port=0, open_browser=False, new_server=True)

    # Now call run() again pointing at the same host/port.
    attached = begira.run(host=server.host, port=server.port, open_browser=False)

    # Attached instance should be a client (not a second server process).
    from begira.client import BegiraClient

    assert isinstance(attached, BegiraClient)
    assert attached.base_url.rstrip("/") == f"http://{server.host}:{server.port}"


def test_run_new_server_forces_start_even_if_env_url_is_set() -> None:
    import os

    import begira

    # Start a server and point BEGIRA_URL at it.
    s1 = begira.run(host="127.0.0.1", port=0, open_browser=False, new_server=True)

    os.environ["BEGIRA_URL"] = f"http://{s1.host}:{s1.port}"
    try:
        # new_server=True should ignore BEGIRA_URL and start a fresh server.
        s2 = begira.run(host="127.0.0.1", port=0, open_browser=False, new_server=True)
    finally:
        os.environ.pop("BEGIRA_URL", None)

    from begira.runner import BegiraServer

    assert isinstance(s2, BegiraServer)
    assert (s2.host, s2.port) != (s1.host, s1.port)
