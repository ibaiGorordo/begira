import time
from pathlib import Path
import begira
from begira.io.ply import load_ply_gaussians

def main() -> None:
    # Use a different port to avoid conflicts if another instance is running
    client = begira.run(port=57793)

    assets_dir = Path(__file__).resolve().parent / "assets"
    gs_path = assets_dir / "gaussians.ply"
    
    if not gs_path.exists():
        print(f"File not found: {gs_path}")
        return

    print(f"Loading {gs_path}...")
    gs_data = load_ply_gaussians(str(gs_path))
    
    print(f"Logging gaussians (count={gs_data.positions.shape[0]})...")
    client.log_gaussians("gaussians_model", gs_data)

    print("Gaussians logged. Check your browser.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
