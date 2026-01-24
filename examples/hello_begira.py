import time
import numpy as np
import begira


def main() -> None:
    server = begira.run(port=0, open_browser=True)

    n = 5000
    points1 = np.random.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    server.log_points("points1", points1)


    points2 = np.random.randn(n, 3).astype(np.float32)
    points2 /= np.linalg.norm(points2, axis=1, keepdims=True)

    server.log_points("points2", points2, point_size=0.08)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
