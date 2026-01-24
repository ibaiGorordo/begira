import time
from pathlib import Path

import begira
from begira.ply import load_ply_gaussians


def main() -> None:
    client = begira.run(port=57792)

    assets_dir = Path(__file__).resolve().parent / "assets"
    gs0 = load_ply_gaussians(str(assets_dir / "object_0.ply"))
    client.log_points("gaussians0", gs0.positions*3, gs0.colors_rgb8)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
