import argparse
import glob
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def sh_l1_basis(directions: torch.Tensor):
    return torch.cat([torch.ones((directions.shape[0], 1), dtype=directions.dtype), directions], dim=-1)


def solve_coefficients(basis: torch.Tensor, values: torch.Tensor, ridge: float):
    lhs = basis.T @ basis
    lhs = lhs + ridge * torch.eye(lhs.shape[0], dtype=lhs.dtype)
    rhs = basis.T @ values
    return torch.linalg.solve(lhs, rhs)


def load_triangle_centers(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        triangles = torch.from_numpy(np.asarray(f["triangles"], dtype=np.float32))
    return triangles.mean(dim=1)


def fit_scene(args):
    centers = load_triangle_centers(args.h5_path)
    gt_paths = sorted(glob.glob(args.gt_glob))
    if args.train_modulo > 0:
        gt_paths = [
            path
            for path in gt_paths
            if (int(os.path.splitext(os.path.basename(path))[0].split("_")[-1]) % args.train_modulo) != args.test_remainder
        ]
    if not gt_paths:
        raise FileNotFoundError(f"No train GT matched {args.gt_glob}")

    with np.load(gt_paths[0]) as first:
        num_triangles = int(first["tri_valid_mask"].shape[0])

    obs_counts = torch.zeros((num_triangles,), dtype=torch.int64)
    basis_samples = [[] for _ in range(num_triangles)]
    direct_samples = [[] for _ in range(num_triangles)]
    indirect_samples = [[] for _ in range(num_triangles)]

    for gt_path in gt_paths:
        with np.load(gt_path) as gt:
            valid = np.asarray(gt["tri_valid_mask"], dtype=bool)
            direct = torch.from_numpy(np.asarray(gt["L_direct_tri"], dtype=np.float32))
            indirect = torch.from_numpy(np.asarray(gt["L_indirect_tri"], dtype=np.float32))
            c2w = np.asarray(gt["c2w"], dtype=np.float32)
        valid_ids = np.where(valid)[0]
        if valid_ids.size == 0:
            continue
        camera_pos = torch.from_numpy(c2w[:3, 3].astype(np.float32))
        tri_ids = torch.from_numpy(valid_ids).long()
        dirs = F.normalize(camera_pos[None] - centers[tri_ids], dim=-1)
        basis = sh_l1_basis(dirs)
        for local_idx, tri_id in enumerate(valid_ids.tolist()):
            basis_samples[tri_id].append(basis[local_idx])
            direct_samples[tri_id].append(direct[tri_id])
            indirect_samples[tri_id].append(indirect[tri_id])
            obs_counts[tri_id] += 1

    full_direct = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
    full_indirect = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
    residual = torch.zeros((num_triangles,), dtype=torch.float32)
    fit_mask = obs_counts >= args.min_views
    fitted_ids = torch.where(fit_mask)[0]

    for tri_id in fitted_ids.tolist():
        basis = torch.stack(basis_samples[tri_id]).float()
        direct = torch.stack(direct_samples[tri_id]).float()
        indirect = torch.stack(indirect_samples[tri_id]).float()
        full_direct[tri_id] = solve_coefficients(basis, direct, args.ridge)
        full_indirect[tri_id] = solve_coefficients(basis, indirect, args.ridge)
        pred_direct = basis @ full_direct[tri_id]
        pred_indirect = basis @ full_indirect[tri_id]
        residual[tri_id] = 0.5 * (F.mse_loss(pred_direct, direct) + F.mse_loss(pred_indirect, indirect))

    compact_ids = fitted_ids.long()
    target = {
        "h5_path": args.h5_path.replace("\\", "/"),
        "scene": args.scene,
        "triangle_id": compact_ids,
        "direct_sh_l1": full_direct[compact_ids],
        "indirect_sh_l1": full_indirect[compact_ids],
        "total_sh_l1": full_direct[compact_ids] + full_indirect[compact_ids],
        "fit_view_count": obs_counts[compact_ids],
        "fit_residual": residual[compact_ids],
        "valid_mask": fit_mask,
        "full_direct_sh_l1": full_direct,
        "full_indirect_sh_l1": full_indirect,
        "obs_counts": obs_counts,
        "basis": "1,x,y,z",
        "min_views": args.min_views,
        "ridge": args.ridge,
        "train_gt_paths": [path.replace("\\", "/") for path in gt_paths],
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(target, args.output_path)
    print(
        f"saved {args.output_path} | scene={args.scene} | views={len(gt_paths)} | "
        f"fitted_triangles={int(fit_mask.sum())}/{num_triangles}"
    )


def main():
    parser = argparse.ArgumentParser(description="Fit Stage 0 SH L=1 targets directly from GT NPZ files")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--h5_path", required=True)
    parser.add_argument("--gt_glob", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--min_views", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--train_modulo", type=int, default=3)
    parser.add_argument("--test_remainder", type=int, default=0)
    args = parser.parse_args()
    fit_scene(args)


if __name__ == "__main__":
    main()
