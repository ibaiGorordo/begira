import time
from pathlib import Path

import numpy as np

import begira
from begira.ply import load_ply_gaussians


def main() -> None:
    client = begira.run(port=57793)

    client.set_coordinate_convention(begira.CoordinateConvention.Z_UP)

    assets_dir = Path(__file__).resolve().parent / "assets"
    gs = load_ply_gaussians(str(assets_dir / "gaussians.ply"))
    client.log_gaussians("gaussians", gs)
    client.log_points("points", gs.positions*np.array([1, -1, 1])+np.array([0, -5, 0]), gs.colors_rgb8, point_size=0.025)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
