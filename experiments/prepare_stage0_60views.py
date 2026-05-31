import argparse
import json
import math
import os
from pathlib import Path


ELEVATIONS_DEG = [10.0, 20.0, 30.0, 40.0]
AZIMUTH_COUNT = 15


def orbit_camera(base_camera: dict, elevation_deg: float, azimuth_deg: float):
    look_at = base_camera.get("look_at", [0.0, 0.0, 0.0])
    base_position = base_camera.get("position", [0.0, -2.0, 0.0])
    dx = float(base_position[0]) - float(look_at[0])
    dy = float(base_position[1]) - float(look_at[1])
    dz = float(base_position[2]) - float(look_at[2])
    radius = math.sqrt(dx * dx + dy * dy + dz * dz)
    if radius <= 1e-6:
        radius = 2.0

    elev = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    xy = radius * math.cos(elev)
    position = [
        float(look_at[0]) + xy * math.sin(az),
        float(look_at[1]) - xy * math.cos(az),
        float(look_at[2]) + radius * math.sin(elev),
    ]
    return {
        "position": position,
        "look_at": look_at,
        "up": base_camera.get("up", [0.0, 0.0, 1.0]),
        "fov": base_camera.get("fov", 37.5),
    }


def make_cameras(base_camera: dict):
    cameras = []
    for elevation in ELEVATIONS_DEG:
        for az_idx in range(AZIMUTH_COUNT):
            azimuth = 360.0 * az_idx / AZIMUTH_COUNT
            cameras.append(orbit_camera(base_camera, elevation, azimuth))
    return cameras


def rewrite_mesh_paths(scene: dict, scene_config_dir: Path, output_scene_dir: Path):
    examples_dir = scene_config_dir.resolve()
    for obj in scene["objects"].values():
        mesh_path = Path(obj["mesh_path"])
        abs_mesh_path = (examples_dir / mesh_path).resolve()
        obj["mesh_path"] = os.path.relpath(abs_mesh_path, output_scene_dir.resolve()).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 0 60-view configs for examples scenes")
    parser.add_argument("--examples_dir", default="examples")
    parser.add_argument("--output_root", default="output/stage0_examples_60views")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scene_paths = sorted(path for path in examples_dir.glob("*.json"))
    if args.limit is not None:
        scene_paths = scene_paths[: args.limit]

    manifest = []
    for scene_path in scene_paths:
        scene_name = scene_path.stem
        output_scene_dir = output_root / scene_name
        output_scene_dir.mkdir(parents=True, exist_ok=True)
        with open(scene_path, "r") as f:
            scene = json.load(f)

        base_camera = scene["cameras"][0] if scene.get("cameras") else {}
        scene["cameras"] = make_cameras(base_camera)
        rewrite_mesh_paths(scene, examples_dir, output_scene_dir)

        config_path = output_scene_dir / f"{scene_name}_60views.json"
        with open(config_path, "w") as f:
            json.dump(scene, f, indent=2)

        manifest.append(
            {
                "scene": scene_name,
                "source_config": str(scene_path).replace("\\", "/"),
                "config": str(config_path).replace("\\", "/"),
                "mesh": str(output_scene_dir / f"{scene_name}.obj").replace("\\", "/"),
                "h5": str(output_scene_dir / f"{scene_name}_input.h5").replace("\\", "/"),
                "gt_dir": str(output_scene_dir / "stage0_gt").replace("\\", "/"),
                "views": len(scene["cameras"]),
            }
        )

    manifest_path = output_root / "manifest_60views.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "elevations_deg": ELEVATIONS_DEG,
                "azimuth_count": AZIMUTH_COUNT,
                "view_count": len(ELEVATIONS_DEG) * AZIMUTH_COUNT,
                "scenes": manifest,
            },
            f,
            indent=2,
        )
    print(f"prepared {len(manifest)} scenes at {output_root}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
