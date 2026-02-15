from __future__ import annotations

import argparse

from .runtime.server import run


def main() -> None:
    p = argparse.ArgumentParser(prog="begira", description="begira: minimal point cloud viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    srv = run(host=args.host, port=args.port, open_browser=not args.no_browser)
    print(srv.url)

    # Block forever (so it behaves like a normal CLI server)
    import time

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
