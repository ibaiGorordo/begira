import time
from pathlib import Path

import begira
from begira.io.ply import load_ply_gaussians


def main() -> None:
    client = begira.run(port=57793)

    assets_dir = Path(__file__).resolve().parent / "assets"
    gs = load_ply_gaussians(str(assets_dir / "gaussians.ply"))
    gs_obj = client.log_gaussians("gaussians", gs)

    main_camera = client.log_camera(
        "main_camera",
        fov=60.0,
    )

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
