import argparse
import json
import os

import h5py
import numpy as np
import torch
import trimesh


def main():
    parser = argparse.ArgumentParser(description="Debug Stage 0 triangle/cache alignment")
    parser.add_argument("--scene_config", required=True)
    parser.add_argument("--h5_path", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--target_path", default=None)
    parser.add_argument("--feature_cache_path", default=None)
    parser.add_argument("--gt_path", default=None)
    args = parser.parse_args()

    with open(args.scene_config, "r") as f:
        config = json.load(f)
    with h5py.File(args.h5_path, "r") as f:
        h5_triangles = np.asarray(f["triangles"], dtype=np.float32)

    with open(args.mapping_path, "r") as f:
        mapping = json.load(f)

    mapping_counts = {}
    for item in mapping["mapping"]:
        mapping_counts[item["source_object"]] = mapping_counts.get(item["source_object"], 0) + 1

    offset = 0
    order_ok = True
    print(f"h5_triangles={len(h5_triangles)} mapping_triangles={mapping['n_triangles']}")
    print("object,offset,faces,mapping_count,max_abs_h5_vs_split_obj")
    for obj_key in config["objects"].keys():
        mesh_path = os.path.join(args.split_dir, f"{obj_key}.obj")
        mesh = trimesh.load(mesh_path, process=False, force="mesh")
        triangles = np.asarray(mesh.triangles, dtype=np.float32)
        n = int(triangles.shape[0])
        h5_slice = h5_triangles[offset : offset + n]
        max_abs = float(np.max(np.abs(h5_slice - triangles))) if n else 0.0
        if max_abs > 1e-5:
            order_ok = False
        print(f"{obj_key},{offset},{n},{mapping_counts.get(obj_key, 0)},{max_abs:.8g}")
        offset += n

    print(f"offset_total={offset}")
    print(f"triangle_order_matches_split_objs={order_ok}")

    if args.target_path and args.feature_cache_path:
        target = torch.load(args.target_path, map_location="cpu")
        sample = torch.load(args.feature_cache_path, map_location="cpu")
        print(f"target_h5={target.get('h5_path')}")
        print(f"feature_cache_h5={sample.get('h5_path')}")
        print(f"target_triangles={int(target['valid_mask'].shape[0])}")
        print(f"feature_triangles={int(sample['latents'].shape[0])}")
        print(f"fitted_triangles={int(target['valid_mask'].sum())}")

    if args.target_path and args.gt_path:
        target = torch.load(args.target_path, map_location="cpu")
        valid_mask = target["valid_mask"].cpu().numpy().astype(bool)
        with np.load(args.gt_path) as gt:
            tri_ids = np.asarray(gt["triangle_id_buffer"], dtype=np.int64)
            valid_pixels = tri_ids >= 0
            in_range = valid_pixels & (tri_ids < valid_mask.shape[0])
            fitted_pixels = in_range & valid_mask[tri_ids.clip(0, valid_mask.shape[0] - 1)]
            visible_ids = np.unique(tri_ids[valid_pixels])
            fitted_visible_ids = [tri_id for tri_id in visible_ids if 0 <= tri_id < valid_mask.shape[0] and valid_mask[tri_id]]
        ratio = float(fitted_pixels.sum() / max(1, in_range.sum()))
        print(f"gt_path={args.gt_path}")
        print(f"visible_pixels={int(valid_pixels.sum())}")
        print(f"fitted_pixels={int(fitted_pixels.sum())}")
        print(f"fitted_pixel_ratio={ratio:.6f}")
        print(f"visible_unique_triangles={len(visible_ids)}")
        print(f"fitted_visible_unique_triangles={len(fitted_visible_ids)}")


if __name__ == "__main__":
    main()
