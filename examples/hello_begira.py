import time
from pathlib import Path

import begira
from begira.ply import load_ply_gaussians


def main() -> None:
    client = begira.run(port=57792)

    client.set_coordinate_convention(begira.CoordinateConvention.Z_UP)

    assets_dir = Path(__file__).resolve().parent / "assets"
    gs = load_ply_gaussians(str(assets_dir / "gaussians.ply"))
    client.log_points("gaussians", gs.positions, gs.colors_rgb8)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
