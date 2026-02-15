import time
from pathlib import Path

import begira
import numpy as np
from begira.ply import load_ply_gaussians


def main() -> None:
    client = begira.run(port=57793)

    client.set_coordinate_convention(begira.CoordinateConvention.Z_UP)

    assets_dir = Path(__file__).resolve().parent / "assets"
    gs = load_ply_gaussians(str(assets_dir / "gaussians.ply"))
    gs_obj = client.log_gaussians("gaussians", gs)

    main_camera = client.log_camera(
        "main_camera",
        fov=60.0,
    )

    main_camera.open_view()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
