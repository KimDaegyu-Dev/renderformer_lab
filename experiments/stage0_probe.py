import argparse
import contextlib
import glob
import os
import sys
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import imageio.v3 as iio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderformer import RenderFormerRenderingPipeline


FEATURE_KEYS = ["latents", "material_features", "latent_view_features", "material_view_features"]


@dataclass
class TargetFields:
    triangle_id: str = "triangle_id_buffer"
    direct: str = "I_direct"
    indirect: str = "I_indirect"
    total: str = "I_total"


def load_h5_inputs(path: str):
    with h5py.File(path, "r") as f:
        triangles = torch.from_numpy(np.asarray(f["triangles"], dtype=np.float32))
        texture = torch.from_numpy(np.asarray(f["texture"], dtype=np.float32))
        vn = torch.from_numpy(np.asarray(f["vn"], dtype=np.float32))
    return triangles, texture, vn


def build_material_features(triangles: torch.Tensor, texture: torch.Tensor, vn: torch.Tensor):
    centers = triangles.mean(dim=1)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    areas = 0.5 * torch.linalg.norm(face_cross, dim=-1, keepdim=True)
    face_normals = F.normalize(face_cross, dim=-1)
    vertex_normals = F.normalize(vn.mean(dim=1), dim=-1)
    if texture.dim() == 4:
        texture_features = texture.mean(dim=(-1, -2))
    else:
        texture_features = texture
    return torch.cat(
        [
            centers,
            face_normals,
            vertex_normals,
            torch.log1p(areas),
            texture_features,
        ],
        dim=-1,
    ).float()


def build_geometry_features(triangles: torch.Tensor, vn: torch.Tensor):
    centers = triangles.mean(dim=1)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    face_normals = F.normalize(torch.cross(edge_a, edge_b, dim=-1), dim=-1)
    vertex_normals = F.normalize(vn.mean(dim=1), dim=-1)
    return centers.float(), face_normals.float(), vertex_normals.float()


def load_gt_camera_position(gt_path: str):
    c2w = None
    if gt_path.endswith(".npz"):
        with np.load(gt_path) as f:
            if "c2w" in f:
                c2w = np.asarray(f["c2w"], dtype=np.float32)
    else:
        with h5py.File(gt_path, "r") as f:
            if "c2w" in f:
                c2w = np.asarray(f["c2w"], dtype=np.float32)
    if c2w is None:
        return None
    return torch.from_numpy(c2w[:3, 3].astype(np.float32))


def build_view_dirs(centers: torch.Tensor, gt_path: str):
    camera_pos = load_gt_camera_position(gt_path)
    if camera_pos is None:
        return torch.zeros_like(centers)
    return F.normalize(camera_pos[None] - centers, dim=-1).float()


def ensure_view_axis(array: np.ndarray, color: bool = False) -> np.ndarray:
    if color:
        if array.ndim == 3:
            return array[None]
        return array
    if array.ndim == 2:
        return array[None]
    return array


def triangle_mean_targets(
    h5_path: str,
    num_triangles: int,
    fields: TargetFields,
    min_pixels: int,
):
    with h5py.File(h5_path, "r") as f:
        missing = [name for name in [fields.triangle_id, fields.direct, fields.indirect] if name not in f]
        if missing:
            return None

        tri_ids = ensure_view_axis(np.asarray(f[fields.triangle_id]))
        direct = ensure_view_axis(np.asarray(f[fields.direct], dtype=np.float32), color=True)
        indirect = ensure_view_axis(np.asarray(f[fields.indirect], dtype=np.float32), color=True)
        total = (
            ensure_view_axis(np.asarray(f[fields.total], dtype=np.float32), color=True)
            if fields.total in f
            else direct + indirect
        )

    sums_direct = np.zeros((num_triangles, 3), dtype=np.float64)
    sums_indirect = np.zeros((num_triangles, 3), dtype=np.float64)
    sums_total = np.zeros((num_triangles, 3), dtype=np.float64)
    counts = np.zeros((num_triangles,), dtype=np.int64)

    for ids, d_img, i_img, t_img in zip(tri_ids, direct, indirect, total):
        flat_ids = ids.reshape(-1).astype(np.int64)
        valid = (flat_ids >= 0) & (flat_ids < num_triangles)
        flat_ids = flat_ids[valid]
        if flat_ids.size == 0:
            continue
        np.add.at(sums_direct, flat_ids, d_img.reshape(-1, 3)[valid])
        np.add.at(sums_indirect, flat_ids, i_img.reshape(-1, 3)[valid])
        np.add.at(sums_total, flat_ids, t_img.reshape(-1, 3)[valid])
        np.add.at(counts, flat_ids, 1)

    denom = np.maximum(counts[:, None], 1)
    target = {
        "direct": torch.from_numpy((sums_direct / denom).astype(np.float32)),
        "indirect": torch.from_numpy((sums_indirect / denom).astype(np.float32)),
        "total": torch.from_numpy((sums_total / denom).astype(np.float32)),
        "target_mask": torch.from_numpy(counts >= min_pixels),
        "pixel_count": torch.from_numpy(counts),
    }
    return target


def load_stage0_targets(
    gt_h5_path: str,
    num_triangles: int,
    fields: TargetFields,
    min_pixels: int,
):
    if gt_h5_path.endswith(".npz"):
        with np.load(gt_h5_path) as f:
            direct = np.asarray(f["L_direct_tri"], dtype=np.float32)
            indirect = np.asarray(f["L_indirect_tri"], dtype=np.float32)
            total = np.asarray(f["L_total_tri"], dtype=np.float32) if "L_total_tri" in f else direct + indirect
            target_mask = np.asarray(f["tri_valid_mask"]).astype(bool)
            pixel_count = np.asarray(f["tri_visible_count"])
        if direct.shape[0] != num_triangles:
            raise ValueError(
                f"GT triangle count mismatch for {gt_h5_path}: "
                f"targets={direct.shape[0]}, RenderFormer input={num_triangles}"
            )
        return {
            "direct": torch.from_numpy(direct),
            "indirect": torch.from_numpy(indirect),
            "total": torch.from_numpy(total),
            "target_mask": torch.from_numpy(target_mask),
            "pixel_count": torch.from_numpy(pixel_count),
        }

    with h5py.File(gt_h5_path, "r") as f:
        if {"L_direct_tri", "L_indirect_tri"}.issubset(f.keys()):
            direct = np.asarray(f["L_direct_tri"], dtype=np.float32)
            indirect = np.asarray(f["L_indirect_tri"], dtype=np.float32)
            total = (
                np.asarray(f["L_total_tri"], dtype=np.float32)
                if "L_total_tri" in f
                else direct + indirect
            )
            target_mask = (
                np.asarray(f["tri_valid_mask"]).astype(bool)
                if "tri_valid_mask" in f
                else np.ones((direct.shape[0],), dtype=bool)
            )
            pixel_count = (
                np.asarray(f["tri_visible_count"])
                if "tri_visible_count" in f
                else target_mask.astype(np.int64)
            )

            if direct.shape[0] != num_triangles:
                raise ValueError(
                    f"GT triangle count mismatch for {gt_h5_path}: "
                    f"targets={direct.shape[0]}, RenderFormer input={num_triangles}"
                )

            return {
                "direct": torch.from_numpy(direct),
                "indirect": torch.from_numpy(indirect),
                "total": torch.from_numpy(total),
                "target_mask": torch.from_numpy(target_mask),
                "pixel_count": torch.from_numpy(pixel_count),
            }

    return triangle_mean_targets(gt_h5_path, num_triangles, fields, min_pixels)


def extract_latents(args):
    device = torch.device(args.device)
    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)
    pipeline.to(device)
    pipeline.model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    fields = TargetFields(args.triangle_id_field, args.direct_field, args.indirect_field, args.total_field)
    h5_paths = sorted(glob.glob(args.h5_glob))
    if not h5_paths:
        raise FileNotFoundError(f"No H5 files matched: {args.h5_glob}")
    gt_paths = sorted(glob.glob(args.gt_h5_glob)) if args.gt_h5_glob else []
    if gt_paths and len(gt_paths) not in [1, len(h5_paths)] and len(h5_paths) != 1:
        raise ValueError("--gt_h5_glob must match one GT, one GT per H5, or multiple GT views for a single H5")
    duplicate_basenames = len({os.path.basename(path) for path in h5_paths}) != len(h5_paths)

    for index, h5_path in enumerate(h5_paths):
        triangles, texture, vn = load_h5_inputs(h5_path)
        num_triangles = triangles.shape[0]
        mask = torch.ones(num_triangles, dtype=torch.bool)
        centers, face_normals, vertex_normals = build_geometry_features(triangles, vn)

        if pipeline.config.texture_encode_patch_size == 1 and texture.dim() == 4:
            texture = texture[:, :, 0, 0]
        if not pipeline.config.use_ldr:
            texture[:, -3:] = torch.log10(texture[:, -3:] + 1.0)
        material_features = build_material_features(triangles, texture, vn)

        amp = (
            torch.autocast(device_type=device.type, dtype=getattr(torch, args.precision))
            if args.precision != "float32"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp:
            latents, valid_mask = pipeline.model.extract_view_independent_latents(
                triangles.reshape(1, -1, 9).to(device),
                texture.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
                vn.reshape(1, -1, 9).to(device),
            )

        if not gt_paths:
            sample_gt_paths = [h5_path]
        elif len(h5_paths) == 1:
            sample_gt_paths = gt_paths
        elif len(gt_paths) == 1:
            sample_gt_paths = [gt_paths[0]]
        else:
            sample_gt_paths = [gt_paths[index]]

        for gt_path in sample_gt_paths:
            view_dirs = build_view_dirs(centers, gt_path)
            latent_features = latents[0].float().cpu()
            sample = {
                "h5_path": h5_path,
                "gt_path": gt_path,
                "latents": latent_features,
                "material_features": material_features.cpu(),
                "triangle_centers": centers.cpu(),
                "triangle_normals": vertex_normals.cpu(),
                "face_normals": face_normals.cpu(),
                "view_dirs": view_dirs.cpu(),
                "latent_view_features": torch.cat([latent_features, vertex_normals.cpu(), view_dirs.cpu()], dim=-1),
                "material_view_features": torch.cat([material_features.cpu(), vertex_normals.cpu(), view_dirs.cpu()], dim=-1),
                "valid_mask": valid_mask[0].cpu(),
            }
            targets = load_stage0_targets(gt_path, num_triangles, fields, args.min_pixels)
            if targets is not None:
                sample.update(targets)

            stem = os.path.splitext(os.path.basename(h5_path))[0]
            if duplicate_basenames:
                parent = os.path.basename(os.path.dirname(h5_path))
                stem = f"{parent}_{stem}"
            if len(sample_gt_paths) > 1:
                gt_stem = os.path.splitext(os.path.basename(gt_path))[0]
                stem = f"{stem}_{gt_stem}"
            out_name = stem + ".pt"
            out_path = os.path.join(args.output_dir, out_name)
            torch.save(sample, out_path)
            has_targets = "direct" in sample and bool(sample["target_mask"].any())
            print(f"saved {out_path} | triangles={num_triangles} | targets={has_targets}")


class ProbeDataset(Dataset):
    def __init__(self, cache_glob: str, feature_key: str = "latents"):
        paths = sorted(glob.glob(cache_glob))
        if not paths:
            raise FileNotFoundError(f"No latent caches matched: {cache_glob}")
        xs, ys, totals = [], [], []
        for path in paths:
            sample = torch.load(path, map_location="cpu")
            if not {"direct", "indirect", "target_mask"}.issubset(sample):
                print(f"skip {path}: missing direct/indirect targets")
                continue
            mask = sample["valid_mask"] & sample["target_mask"]
            if not bool(mask.any()):
                print(f"skip {path}: no supervised triangles")
                continue
            if feature_key not in sample:
                raise KeyError(f"{path} does not contain feature key '{feature_key}'")
            xs.append(sample[feature_key][mask])
            ys.append(torch.cat([sample["direct"][mask], sample["indirect"][mask]], dim=-1))
            totals.append(sample["total"][mask])
        if not xs:
            raise RuntimeError("No usable supervised triangles found.")
        self.x = torch.cat(xs, dim=0)
        self.y = torch.cat(ys, dim=0)
        self.total = torch.cat(totals, dim=0)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.total[index]


class RadianceProbe(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


class ShProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target).clamp_min(eps)
    max_val = target.max().clamp_min(1.0)
    return 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse)


def read_gt_bundle(path: str):
    if path.endswith(".npz"):
        with np.load(path) as f:
            return {key: np.asarray(f[key]) for key in f.files}

    with h5py.File(path, "r") as f:
        return {key: np.asarray(f[key]) for key in f.keys()}


def gather_triangle_image(triangle_values: np.ndarray, triangle_id_buffer: np.ndarray):
    image = np.zeros((*triangle_id_buffer.shape, triangle_values.shape[-1]), dtype=np.float32)
    valid = (triangle_id_buffer >= 0) & (triangle_id_buffer < triangle_values.shape[0])
    image[valid] = triangle_values[triangle_id_buffer[valid]]
    return image


def ldr_tonemap(image: np.ndarray, mode: str, exposure: float = 1.0):
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image * exposure, 0.0)
    if mode == "clip":
        mapped = np.clip(image, 0.0, 1.0)
    elif mode == "reinhard":
        mapped = image / (1.0 + image)
    elif mode == "percentile":
        denom = np.percentile(image[image > 0], 99.0) if np.any(image > 0) else 1.0
        mapped = np.clip(image / max(float(denom), 1e-6), 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported tonemap: {mode}")
    return (mapped * 255.0 + 0.5).astype(np.uint8)


def save_image_outputs(base_path: str, image: np.ndarray, tonemap: str, exposure: float, save_exr: bool):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    iio.imwrite(base_path + ".png", ldr_tonemap(image, tonemap, exposure))
    if save_exr:
        try:
            iio.imwrite(base_path + ".exr", image.astype(np.float32))
        except Exception as exc:
            print(f"[WARN] Failed to save {base_path}.exr: {exc}")


def save_diff_image_outputs(base_path: str, diff_image: np.ndarray, scale: float, save_exr: bool):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    image = np.nan_to_num(diff_image.astype(np.float32) * scale, nan=0.0, posinf=0.0, neginf=0.0)
    signed_png = np.clip(0.5 + image, 0.0, 1.0)
    iio.imwrite(base_path + ".png", (signed_png * 255.0 + 0.5).astype(np.uint8))
    if save_exr:
        try:
            iio.imwrite(base_path + ".exr", diff_image.astype(np.float32))
        except Exception as exc:
            print(f"[WARN] Failed to save {base_path}.exr: {exc}")


def make_contact_sheet(images: list[np.ndarray], labels: list[str], tonemap: str, exposure: float):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    rendered = [ldr_tonemap(image, tonemap, exposure) for image in images]
    h, w = rendered[0].shape[:2]
    label_h = 20
    sheet = Image.new("RGB", (w * len(rendered), h + label_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for idx, (img, label) in enumerate(zip(rendered, labels)):
        sheet.paste(Image.fromarray(img), (idx * w, label_h))
        draw.text((idx * w + 4, 4), label, fill=(255, 255, 255))
    return np.asarray(sheet)


def train_probe(args):
    device = torch.device(args.device)
    dataset = ProbeDataset(args.cache_glob, feature_key=args.feature_key)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = RadianceProbe(dataset.x.shape[-1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y, total in loader:
            x, y, total = x.to(device), y.to(device), total.to(device)
            pred = model(x)
            comp_loss = F.l1_loss(pred, y)
            sum_loss = F.l1_loss(pred[:, :3] + pred[:, 3:], total)
            loss = comp_loss + args.lambda_sum * sum_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]

        with torch.no_grad():
            x = dataset.x.to(device)
            y = dataset.y.to(device)
            total = dataset.total.to(device)
            pred = model(x)
            direct_psnr = psnr(pred[:, :3], y[:, :3])
            indirect_psnr = psnr(pred[:, 3:], y[:, 3:])
            total_psnr = psnr(pred[:, :3] + pred[:, 3:], total)
        print(
            f"epoch {epoch + 1:03d} "
            f"loss={total_loss / len(dataset):.6f} "
            f"direct_psnr={direct_psnr:.2f} "
            f"indirect_psnr={indirect_psnr:.2f} "
            f"total_psnr={total_psnr:.2f}"
        )

    if args.output_model:
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
        torch.save({"model": model.state_dict(), "latent_dim": dataset.x.shape[-1]}, args.output_model)
        print(f"saved {args.output_model}")


def predict_probe_triangles(cache_path: str, probe_path: str, device: torch.device, hidden_dim: int, feature_key: str):
    sample = torch.load(cache_path, map_location="cpu")
    checkpoint = torch.load(probe_path, map_location="cpu")
    if feature_key not in sample:
        raise KeyError(f"{cache_path} does not contain feature key '{feature_key}'")
    latent_dim = int(checkpoint.get("latent_dim", sample[feature_key].shape[-1]))

    model = RadianceProbe(latent_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        pred = model(sample[feature_key].to(device)).cpu().numpy().astype(np.float32)
    return sample, pred[:, :3], pred[:, 3:]


def eval_gather(args):
    device = torch.device(args.device)
    _, pred_direct_tri, pred_indirect_tri = predict_probe_triangles(
        args.cache_path,
        args.probe_path,
        device,
        args.hidden_dim,
        args.feature_key,
    )
    pred_total_tri = pred_direct_tri + pred_indirect_tri

    gt = read_gt_bundle(args.gt_path)
    triangle_id_buffer = gt["triangle_id_buffer"].astype(np.int64)
    valid_pixel_mask = gt.get("valid_pixel_mask", triangle_id_buffer >= 0).astype(bool)

    pred_direct_img = gather_triangle_image(pred_direct_tri, triangle_id_buffer)
    pred_indirect_img = gather_triangle_image(pred_indirect_tri, triangle_id_buffer)
    pred_total_img = gather_triangle_image(pred_total_tri, triangle_id_buffer)

    gt_direct_img = gt["I_direct"].astype(np.float32)
    gt_indirect_img = gt["I_indirect"].astype(np.float32)
    gt_total_img = gt["I_total"].astype(np.float32)
    gt_recomposed_img = gt_direct_img + gt_indirect_img

    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {
        "pred_direct": pred_direct_img,
        "pred_indirect": pred_indirect_img,
        "pred_total": pred_total_img,
        "gt_direct": gt_direct_img,
        "gt_indirect": gt_indirect_img,
        "gt_total": gt_total_img,
        "gt_recomposed": gt_recomposed_img,
        "absdiff_direct": np.abs(pred_direct_img - gt_direct_img),
        "absdiff_indirect": np.abs(pred_indirect_img - gt_indirect_img),
        "absdiff_total_combined": np.abs(pred_total_img - gt_total_img),
        "absdiff_total_recomposed": np.abs(pred_total_img - gt_recomposed_img),
    }
    for name, image in outputs.items():
        save_image_outputs(
            os.path.join(args.output_dir, name),
            image,
            tonemap=args.tonemap,
            exposure=args.exposure,
            save_exr=args.save_exr,
        )

    sheet = make_contact_sheet(
        [gt_direct_img, pred_direct_img, gt_indirect_img, pred_indirect_img, gt_recomposed_img, pred_total_img],
        ["gt direct", "pred direct", "gt indirect", "pred indirect", "gt dir+ind", "pred total"],
        args.tonemap,
        args.exposure,
    )
    if sheet is not None:
        iio.imwrite(os.path.join(args.output_dir, "comparison_sheet.png"), sheet)

    mask = torch.from_numpy(valid_pixel_mask)
    metrics = {
        "direct_psnr_img": float(psnr(torch.from_numpy(pred_direct_img)[mask], torch.from_numpy(gt_direct_img)[mask])),
        "indirect_psnr_img": float(psnr(torch.from_numpy(pred_indirect_img)[mask], torch.from_numpy(gt_indirect_img)[mask])),
        "total_psnr_vs_combined_img": float(psnr(torch.from_numpy(pred_total_img)[mask], torch.from_numpy(gt_total_img)[mask])),
        "total_psnr_vs_direct_plus_indirect_img": float(psnr(torch.from_numpy(pred_total_img)[mask], torch.from_numpy(gt_recomposed_img)[mask])),
        **component_leakage_metrics(pred_direct_img, pred_indirect_img, gt_direct_img, gt_indirect_img, valid_pixel_mask),
        "valid_pixels": int(valid_pixel_mask.sum()),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"saved gathered evaluation outputs to {args.output_dir}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def image_metrics_for_triangles(pred_direct_tri, pred_indirect_tri, gt, valid_pixel_mask):
    triangle_id_buffer = gt["triangle_id_buffer"].astype(np.int64)
    pred_direct_img = gather_triangle_image(pred_direct_tri, triangle_id_buffer)
    pred_indirect_img = gather_triangle_image(pred_indirect_tri, triangle_id_buffer)
    pred_total_img = pred_direct_img + pred_indirect_img
    gt_direct_img = gt["I_direct"].astype(np.float32)
    gt_indirect_img = gt["I_indirect"].astype(np.float32)
    gt_recomposed_img = gt_direct_img + gt_indirect_img
    mask = torch.from_numpy(valid_pixel_mask)
    return {
        "direct_psnr_img": float(psnr(torch.from_numpy(pred_direct_img)[mask], torch.from_numpy(gt_direct_img)[mask])),
        "indirect_psnr_img": float(psnr(torch.from_numpy(pred_indirect_img)[mask], torch.from_numpy(gt_indirect_img)[mask])),
        "total_psnr_vs_direct_plus_indirect_img": float(psnr(torch.from_numpy(pred_total_img)[mask], torch.from_numpy(gt_recomposed_img)[mask])),
        **component_leakage_metrics(pred_direct_img, pred_indirect_img, gt_direct_img, gt_indirect_img, valid_pixel_mask),
    }


def component_leakage_metrics(pred_direct_img, pred_indirect_img, gt_direct_img, gt_indirect_img, valid_pixel_mask):
    mask = torch.from_numpy(valid_pixel_mask)
    pred_direct = torch.from_numpy(pred_direct_img)[mask]
    pred_indirect = torch.from_numpy(pred_indirect_img)[mask]
    gt_direct = torch.from_numpy(gt_direct_img)[mask]
    gt_indirect = torch.from_numpy(gt_indirect_img)[mask]

    direct_to_direct = psnr(pred_direct, gt_direct)
    direct_to_indirect = psnr(pred_direct, gt_indirect)
    indirect_to_indirect = psnr(pred_indirect, gt_indirect)
    indirect_to_direct = psnr(pred_indirect, gt_direct)
    return {
        "direct_cross_psnr_img": float(direct_to_indirect),
        "indirect_cross_psnr_img": float(indirect_to_direct),
        "direct_leakage_margin_db": float(direct_to_direct - direct_to_indirect),
        "indirect_leakage_margin_db": float(indirect_to_indirect - indirect_to_direct),
    }


def train_global_mean_targets(cache_glob: str):
    paths = sorted(glob.glob(cache_glob))
    if not paths:
        raise FileNotFoundError(f"No train mean caches matched: {cache_glob}")

    direct_sum = torch.zeros(3, dtype=torch.float64)
    indirect_sum = torch.zeros(3, dtype=torch.float64)
    weight_sum = torch.tensor(0.0, dtype=torch.float64)
    for path in paths:
        sample = torch.load(path, map_location="cpu")
        if not {"direct", "indirect", "target_mask", "pixel_count"}.issubset(sample):
            print(f"skip train mean source {path}: missing targets")
            continue
        mask = sample["valid_mask"] & sample["target_mask"]
        if not bool(mask.any()):
            continue
        weights = sample["pixel_count"][mask].double().clamp_min(1.0)
        direct_sum += (sample["direct"][mask].double() * weights[:, None]).sum(dim=0)
        indirect_sum += (sample["indirect"][mask].double() * weights[:, None]).sum(dim=0)
        weight_sum += weights.sum()

    if float(weight_sum) <= 0.0:
        raise RuntimeError(f"No usable train targets found for mean baseline: {cache_glob}")

    direct = (direct_sum / weight_sum).float().numpy()
    indirect = (indirect_sum / weight_sum).float().numpy()
    return direct, indirect


def eval_compare(args):
    device = torch.device(args.device)
    sample, latent_direct, latent_indirect = predict_probe_triangles(
        args.cache_path,
        args.latent_probe_path,
        device,
        args.hidden_dim,
        "latents",
    )
    _, material_direct, material_indirect = predict_probe_triangles(
        args.cache_path,
        args.material_probe_path,
        device,
        args.hidden_dim,
        "material_features",
    )
    if args.mean_cache_glob:
        mean_name = "train_global_mean"
        direct_rgb, indirect_rgb = train_global_mean_targets(args.mean_cache_glob)
    else:
        mean_name = "per_scene_triangle_mean"
        supervised = sample["valid_mask"] & sample["target_mask"]
        direct_rgb = sample["direct"][supervised].mean(dim=0).numpy()
        indirect_rgb = sample["indirect"][supervised].mean(dim=0).numpy()
    direct_mean = direct_rgb[None].repeat(sample["direct"].shape[0], axis=0)
    indirect_mean = indirect_rgb[None].repeat(sample["indirect"].shape[0], axis=0)

    gt = read_gt_bundle(args.gt_path)
    valid_pixel_mask = gt.get("valid_pixel_mask", gt["triangle_id_buffer"] >= 0).astype(bool)
    rows = {
        "renderformer_latent": image_metrics_for_triangles(latent_direct, latent_indirect, gt, valid_pixel_mask),
        "material_only": image_metrics_for_triangles(material_direct, material_indirect, gt, valid_pixel_mask),
        mean_name: image_metrics_for_triangles(direct_mean.astype(np.float32), indirect_mean.astype(np.float32), gt, valid_pixel_mask),
    }
    if args.latent_view_probe_path:
        _, latent_view_direct, latent_view_indirect = predict_probe_triangles(
            args.cache_path,
            args.latent_view_probe_path,
            device,
            args.hidden_dim,
            "latent_view_features",
        )
        rows["renderformer_latent_view"] = image_metrics_for_triangles(latent_view_direct, latent_view_indirect, gt, valid_pixel_mask)
    if args.material_view_probe_path:
        _, material_view_direct, material_view_indirect = predict_probe_triangles(
            args.cache_path,
            args.material_view_probe_path,
            device,
            args.hidden_dim,
            "material_view_features",
        )
        rows["material_view"] = image_metrics_for_triangles(material_view_direct, material_view_indirect, gt, valid_pixel_mask)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "baseline_comparison_metrics.csv")
    with open(metrics_path, "w") as f:
        metric_keys = list(next(iter(rows.values())).keys())
        f.write("method," + ",".join(metric_keys) + "\n")
        for method, metrics in rows.items():
            f.write(method + "," + ",".join(str(metrics[key]) for key in metric_keys) + "\n")

    print(f"saved baseline comparison to {metrics_path}")
    for method, metrics in rows.items():
        print(method)
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def sh_l1_basis(directions: torch.Tensor):
    dirs = F.normalize(directions.float(), dim=-1)
    return torch.cat([torch.ones((dirs.shape[0], 1), dtype=dirs.dtype), dirs], dim=-1)


def eval_sh_l1(coeffs: torch.Tensor, directions: torch.Tensor):
    basis = sh_l1_basis(directions)
    return torch.einsum("...b,...bc->...c", basis, coeffs)


def fit_sh_l1(args):
    paths = sorted(glob.glob(args.cache_glob))
    if not paths:
        raise FileNotFoundError(f"No caches matched: {args.cache_glob}")

    groups = {}
    for path in paths:
        sample = torch.load(path, map_location="cpu")
        h5_path = sample.get("h5_path", os.path.splitext(os.path.basename(path))[0])
        groups.setdefault(h5_path, []).append((path, sample))

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    for h5_path, entries in groups.items():
        num_triangles = entries[0][1]["direct"].shape[0]
        direct_coeffs = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
        indirect_coeffs = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
        fit_mask = torch.zeros((num_triangles,), dtype=torch.bool)
        obs_counts = torch.zeros((num_triangles,), dtype=torch.long)
        residuals = torch.zeros((num_triangles,), dtype=torch.float32)

        for tri_idx in range(num_triangles):
            view_dirs, direct, indirect = [], [], []
            for _, sample in entries:
                mask = sample["valid_mask"] & sample["target_mask"]
                if tri_idx >= mask.shape[0] or not bool(mask[tri_idx]):
                    continue
                view_dirs.append(sample["view_dirs"][tri_idx])
                direct.append(sample["direct"][tri_idx])
                indirect.append(sample["indirect"][tri_idx])
            obs_counts[tri_idx] = len(view_dirs)
            if len(view_dirs) < args.min_views:
                continue

            basis = sh_l1_basis(torch.stack(view_dirs))
            reg = args.ridge * torch.eye(4, dtype=torch.float32)
            lhs = basis.T @ basis + reg
            direct_coeffs[tri_idx] = torch.linalg.solve(lhs, basis.T @ torch.stack(direct))
            indirect_coeffs[tri_idx] = torch.linalg.solve(lhs, basis.T @ torch.stack(indirect))
            direct_fit = eval_sh_l1(direct_coeffs[tri_idx].expand(basis.shape[0], -1, -1), torch.stack(view_dirs))
            indirect_fit = eval_sh_l1(indirect_coeffs[tri_idx].expand(basis.shape[0], -1, -1), torch.stack(view_dirs))
            residuals[tri_idx] = 0.5 * (
                F.mse_loss(direct_fit, torch.stack(direct)) + F.mse_loss(indirect_fit, torch.stack(indirect))
            )
            fit_mask[tri_idx] = True

        scene_name = os.path.basename(os.path.dirname(h5_path)) or os.path.splitext(os.path.basename(h5_path))[0]
        out_path = args.output_path if args.output_path and len(groups) == 1 else os.path.join(args.output_dir, f"{scene_name}_sh_l1.pt")
        valid_ids = torch.nonzero(fit_mask, as_tuple=False).flatten()
        compact = {
            "h5_path": h5_path,
            "triangle_id": valid_ids,
            "direct_sh_l1": direct_coeffs[fit_mask],
            "indirect_sh_l1": indirect_coeffs[fit_mask],
            "total_sh_l1": direct_coeffs[fit_mask] + indirect_coeffs[fit_mask],
            "fit_view_count": obs_counts[fit_mask],
            "fit_residual": residuals[fit_mask],
            "valid_mask": fit_mask,
            "full_direct_sh_l1": direct_coeffs,
            "full_indirect_sh_l1": indirect_coeffs,
            "obs_counts": obs_counts,
            "basis": "1,x,y,z",
            "min_views": args.min_views,
            "ridge": args.ridge,
        }
        torch.save(
            compact,
            out_path,
        )
        summary_rows.append((scene_name, len(entries), int(fit_mask.sum()), int(num_triangles)))
        print(f"saved {out_path} | views={len(entries)} | fitted_triangles={int(fit_mask.sum())}/{num_triangles}")

    summary_path = os.path.join(args.output_dir, "sh_l1_summary.csv")
    with open(summary_path, "w") as f:
        f.write("scene,view_caches,fitted_triangles,total_triangles\n")
        for row in summary_rows:
            f.write(",".join(str(value) for value in row) + "\n")
    print(f"saved {summary_path}")


def sh_feature_tensor(sample: dict, feature_key: str):
    if feature_key == "latent_material_features":
        return torch.cat([sample["latents"], sample["material_features"]], dim=-1)
    if feature_key not in sample:
        raise KeyError(f"feature key '{feature_key}' not found in cache")
    return sample[feature_key]


class ShDataset(Dataset):
    def __init__(self, target_path: str, feature_cache_path: str, feature_key: str, normalize: bool = True):
        target = torch.load(target_path, map_location="cpu")
        sample = torch.load(feature_cache_path, map_location="cpu")
        target_h5 = os.path.normpath(str(target.get("h5_path", "")))
        sample_h5 = os.path.normpath(str(sample.get("h5_path", "")))
        if target_h5 and sample_h5 and target_h5 != sample_h5:
            raise ValueError(f"target/cache H5 mismatch: target={target_h5} cache={sample_h5}")
        tri_ids = target["triangle_id"].long()
        features = sh_feature_tensor(sample, feature_key)
        self.x = features[tri_ids].float()
        self.feature_mean = self.x.mean(dim=0, keepdim=True)
        self.feature_std = self.x.std(dim=0, keepdim=True).clamp_min(1e-6)
        if normalize:
            self.x = (self.x - self.feature_mean) / self.feature_std
        direct = target["direct_sh_l1"].reshape(tri_ids.shape[0], -1)
        indirect = target["indirect_sh_l1"].reshape(tri_ids.shape[0], -1)
        self.y = torch.cat([direct, indirect], dim=-1).float()
        self.total = target["total_sh_l1"].reshape(tri_ids.shape[0], -1).float()
        self.h5_path = sample_h5

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.total[index]


def train_sh_probe(args):
    device = torch.device(args.device)
    dataset = ShDataset(args.target_path, args.feature_cache_path, args.feature_key, normalize=not args.no_feature_norm)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = ShProbe(dataset.x.shape[-1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y, total in loader:
            x, y, total = x.to(device), y.to(device), total.to(device)
            pred = model(x)
            comp_loss = F.smooth_l1_loss(pred, y) if args.loss == "smooth_l1" else F.mse_loss(pred, y)
            pred_total = pred[:, :12] + pred[:, 12:]
            sum_loss = F.smooth_l1_loss(pred_total, total) if args.loss == "smooth_l1" else F.mse_loss(pred_total, total)
            loss = comp_loss + args.lambda_sum * sum_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]

        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            with torch.no_grad():
                x = dataset.x.to(device)
                y = dataset.y.to(device)
                total = dataset.total.to(device)
                pred = model(x)
                direct_mse = F.mse_loss(pred[:, :12], y[:, :12])
                indirect_mse = F.mse_loss(pred[:, 12:], y[:, 12:])
                total_mse = F.mse_loss(pred[:, :12] + pred[:, 12:], total)
            print(
                f"epoch {epoch + 1:03d} loss={total_loss / len(dataset):.6f} "
                f"direct_sh_mse={direct_mse:.6f} indirect_sh_mse={indirect_mse:.6f} total_sh_mse={total_mse:.6f}"
            )

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": dataset.x.shape[-1],
            "feature_key": args.feature_key,
            "feature_mean": dataset.feature_mean.squeeze(0),
            "feature_std": dataset.feature_std.squeeze(0),
            "feature_normalized": not args.no_feature_norm,
            "h5_path": dataset.h5_path,
            "hidden_dim": args.hidden_dim,
            "output_dim": 24,
        },
        args.output_model,
    )
    print(f"saved {args.output_model}")


def train_sh_residual_probe(args):
    device = torch.device(args.device)
    dataset = ShDataset(args.target_path, args.feature_cache_path, args.feature_key, normalize=not args.no_feature_norm)
    target, _, _, base_direct, base_indirect = predict_sh_coeffs(
        args.target_path,
        args.feature_cache_path,
        args.material_probe_path,
        device,
    )
    base_y = torch.cat(
        [base_direct.reshape(base_direct.shape[0], -1), base_indirect.reshape(base_indirect.shape[0], -1)],
        dim=-1,
    ).float()
    base_total = (base_direct + base_indirect).reshape(base_direct.shape[0], -1).float()
    dataset.y = dataset.y - base_y
    dataset.total = dataset.total - base_total

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = ShProbe(dataset.x.shape[-1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y, total in loader:
            x, y, total = x.to(device), y.to(device), total.to(device)
            pred = model(x)
            comp_loss = F.smooth_l1_loss(pred, y) if args.loss == "smooth_l1" else F.mse_loss(pred, y)
            pred_total = pred[:, :12] + pred[:, 12:]
            sum_loss = F.smooth_l1_loss(pred_total, total) if args.loss == "smooth_l1" else F.mse_loss(pred_total, total)
            loss = comp_loss + args.lambda_sum * sum_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]

        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            with torch.no_grad():
                x = dataset.x.to(device)
                y = dataset.y.to(device)
                total = dataset.total.to(device)
                pred = model(x)
                direct_mse = F.mse_loss(pred[:, :12], y[:, :12])
                indirect_mse = F.mse_loss(pred[:, 12:], y[:, 12:])
                total_mse = F.mse_loss(pred[:, :12] + pred[:, 12:], total)
            print(
                f"epoch {epoch + 1:03d} loss={total_loss / len(dataset):.6f} "
                f"direct_residual_mse={direct_mse:.6f} indirect_residual_mse={indirect_mse:.6f} "
                f"total_residual_mse={total_mse:.6f}"
            )

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": dataset.x.shape[-1],
            "feature_key": args.feature_key,
            "feature_mean": dataset.feature_mean.squeeze(0),
            "feature_std": dataset.feature_std.squeeze(0),
            "feature_normalized": not args.no_feature_norm,
            "h5_path": dataset.h5_path,
            "hidden_dim": args.hidden_dim,
            "output_dim": 24,
            "residual_base_probe_path": args.material_probe_path,
            "residual_mode": "material_probe",
        },
        args.output_model,
    )
    print(f"saved {args.output_model}")


def predict_sh_coeffs(target_path: str, feature_cache_path: str, probe_path: str, device: torch.device):
    target = torch.load(target_path, map_location="cpu")
    sample = torch.load(feature_cache_path, map_location="cpu")
    checkpoint = torch.load(probe_path, map_location="cpu")
    target_h5 = os.path.normpath(str(target.get("h5_path", "")))
    sample_h5 = os.path.normpath(str(sample.get("h5_path", "")))
    checkpoint_h5 = os.path.normpath(str(checkpoint.get("h5_path", "")))
    if target_h5 and sample_h5 and target_h5 != sample_h5:
        raise ValueError(f"target/cache H5 mismatch: target={target_h5} cache={sample_h5}")
    if checkpoint_h5 and sample_h5 and checkpoint_h5 != sample_h5:
        raise ValueError(f"checkpoint/cache H5 mismatch: checkpoint={checkpoint_h5} cache={sample_h5}")
    feature_key = checkpoint["feature_key"]
    features = sh_feature_tensor(sample, feature_key)
    tri_ids = target["triangle_id"].long()
    pred = predict_sh_tensor_from_checkpoint(features, tri_ids, checkpoint, device)
    return target, sample, tri_ids, pred[:, :12].reshape(-1, 4, 3), pred[:, 12:].reshape(-1, 4, 3)


def predict_sh_tensor_from_checkpoint(features: torch.Tensor, tri_ids: torch.Tensor, checkpoint: dict, device: torch.device):
    model = ShProbe(int(checkpoint["input_dim"]), int(checkpoint.get("hidden_dim", 512))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        x = features[tri_ids].float()
        if checkpoint.get("feature_normalized", False):
            mean = checkpoint["feature_mean"].float()
            std = checkpoint["feature_std"].float().clamp_min(1e-6)
            x = (x - mean) / std
        return model(x.to(device)).cpu()


def predict_sh_coeffs_all(feature_cache_path: str, probe_path: str, device: torch.device):
    sample = torch.load(feature_cache_path, map_location="cpu")
    checkpoint = torch.load(probe_path, map_location="cpu")
    checkpoint_h5 = os.path.normpath(str(checkpoint.get("h5_path", "")))
    sample_h5 = os.path.normpath(str(sample.get("h5_path", "")))
    if checkpoint_h5 and sample_h5 and checkpoint_h5 != sample_h5:
        raise ValueError(f"checkpoint/cache H5 mismatch: checkpoint={checkpoint_h5} cache={sample_h5}")
    features = sh_feature_tensor(sample, checkpoint["feature_key"])
    tri_ids = torch.arange(features.shape[0], dtype=torch.long)
    pred = predict_sh_tensor_from_checkpoint(features, tri_ids, checkpoint, device)
    return pred[:, :12].reshape(-1, 4, 3), pred[:, 12:].reshape(-1, 4, 3)


def clamp_coeffs_to_target_range(coeffs: torch.Tensor, target_coeffs: torch.Tensor, pad_fraction: float):
    target_coeffs = target_coeffs.float()
    lo = target_coeffs.amin(dim=0, keepdim=True)
    hi = target_coeffs.amax(dim=0, keepdim=True)
    pad = (hi - lo).clamp_min(1e-6) * pad_fraction
    return coeffs.float().clamp(lo - pad, hi + pad)


def sh_metrics(pred_direct, pred_indirect, target):
    gt_direct = target["direct_sh_l1"].float()
    gt_indirect = target["indirect_sh_l1"].float()
    pred_total = pred_direct + pred_indirect
    gt_total = gt_direct + gt_indirect
    return {
        "direct_sh_mse": float(F.mse_loss(pred_direct, gt_direct)),
        "indirect_sh_mse": float(F.mse_loss(pred_indirect, gt_indirect)),
        "total_sh_mse": float(F.mse_loss(pred_total, gt_total)),
        "direct_dc_mse": float(F.mse_loss(pred_direct[:, :1], gt_direct[:, :1])),
        "direct_dir_mse": float(F.mse_loss(pred_direct[:, 1:], gt_direct[:, 1:])),
        "indirect_dc_mse": float(F.mse_loss(pred_indirect[:, :1], gt_indirect[:, :1])),
        "indirect_dir_mse": float(F.mse_loss(pred_indirect[:, 1:], gt_indirect[:, 1:])),
    }


def eval_sh_probes(args):
    device = torch.device(args.device)
    probes = [
        ("latent", args.latent_probe_path),
        ("material", args.material_probe_path),
        ("latent_material", args.latent_material_probe_path),
    ]
    rows = {}
    for name, path in probes:
        target, _, _, pred_direct, pred_indirect = predict_sh_coeffs(args.target_path, args.feature_cache_path, path, device)
        rows[name] = sh_metrics(pred_direct, pred_indirect, target)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "sh_l1_probe_metrics.csv")
    metric_keys = list(next(iter(rows.values())).keys())
    with open(out_path, "w") as f:
        f.write("method," + ",".join(metric_keys) + "\n")
        for method, metrics in rows.items():
            f.write(method + "," + ",".join(str(metrics[key]) for key in metric_keys) + "\n")
    print(f"saved {out_path}")
    for method, metrics in rows.items():
        print(method)
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def sh_full_coeffs(target, tri_ids, pred_direct, pred_indirect, fallback_direct=None, fallback_indirect=None):
    num_triangles = int(target["valid_mask"].shape[0])
    if fallback_direct is None:
        full_direct = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
    else:
        full_direct = fallback_direct[:num_triangles].float().clone()
    if fallback_indirect is None:
        full_indirect = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
    else:
        full_indirect = fallback_indirect[:num_triangles].float().clone()
    full_direct[tri_ids] = pred_direct.float()
    full_indirect[tri_ids] = pred_indirect.float()
    return full_direct, full_indirect


def gather_sh_image(coeffs: torch.Tensor, gt: dict, triangle_centers: torch.Tensor):
    tri_ids = gt["triangle_id_buffer"].astype(np.int64)
    valid = (tri_ids >= 0) & (tri_ids < coeffs.shape[0])
    image = np.zeros((*tri_ids.shape, 3), dtype=np.float32)
    if not np.any(valid):
        return image

    c2w = np.asarray(gt["c2w"], dtype=np.float32)
    camera_pos = torch.from_numpy(c2w[:3, 3]).float()
    flat_ids = torch.from_numpy(tri_ids[valid]).long()
    dirs = F.normalize(camera_pos[None] - triangle_centers[flat_ids].float(), dim=-1)
    values = eval_sh_l1(coeffs[flat_ids].float(), dirs).clamp_min(0.0).numpy().astype(np.float32)
    image[valid] = values
    return image


def gather_fit_mask_image(valid_mask: torch.Tensor, gt: dict):
    tri_ids = gt["triangle_id_buffer"].astype(np.int64)
    image = np.zeros((*tri_ids.shape, 3), dtype=np.float32)
    valid = (tri_ids >= 0) & (tri_ids < int(valid_mask.shape[0]))
    if np.any(valid):
        fitted = valid_mask.cpu().numpy().astype(bool)
        image[valid] = fitted[tri_ids[valid], None].astype(np.float32)
    return image


def gather_scalar_image(values: torch.Tensor, gt: dict):
    tri_ids = gt["triangle_id_buffer"].astype(np.int64)
    image = np.zeros((*tri_ids.shape, 3), dtype=np.float32)
    valid = (tri_ids >= 0) & (tri_ids < int(values.shape[0]))
    if np.any(valid):
        scalar = values.cpu().numpy().astype(np.float32)
        image[valid] = scalar[tri_ids[valid], None]
    return image


def gather_coeff_component_image(coeffs: torch.Tensor, component_index: int, gt: dict):
    tri_ids = gt["triangle_id_buffer"].astype(np.int64)
    image = np.zeros((*tri_ids.shape, 3), dtype=np.float32)
    valid = (tri_ids >= 0) & (tri_ids < coeffs.shape[0])
    if np.any(valid):
        values = coeffs[:, component_index].float().numpy()
        image[valid] = values[tri_ids[valid]]
    return image


def gather_sh_negative_mask(coeffs: torch.Tensor, gt: dict, triangle_centers: torch.Tensor):
    tri_ids = gt["triangle_id_buffer"].astype(np.int64)
    image = np.zeros((*tri_ids.shape, 3), dtype=np.float32)
    valid = (tri_ids >= 0) & (tri_ids < coeffs.shape[0])
    if not np.any(valid):
        return image

    c2w = np.asarray(gt["c2w"], dtype=np.float32)
    camera_pos = torch.from_numpy(c2w[:3, 3]).float()
    flat_ids = torch.from_numpy(tri_ids[valid]).long()
    dirs = F.normalize(camera_pos[None] - triangle_centers[flat_ids].float(), dim=-1)
    raw = eval_sh_l1(coeffs[flat_ids].float(), dirs)
    neg = (raw < 0.0).any(dim=-1).numpy().astype(np.float32)
    image[valid] = neg[:, None]
    return image


def save_heldout_sheet(output_dir, view_name, gt_direct, gt_indirect, predictions, tonemap, exposure):
    images = [gt_direct]
    labels = ["gt direct"]
    for method, pred in predictions.items():
        images.append(pred["direct"])
        labels.append(f"{method} direct")
    images.append(gt_indirect)
    labels.append("gt indirect")
    for method, pred in predictions.items():
        images.append(pred["indirect"])
        labels.append(f"{method} indirect")
    gt_total = gt_direct + gt_indirect
    images.append(gt_total)
    labels.append("gt total")
    for method, pred in predictions.items():
        images.append(pred["direct"] + pred["indirect"])
        labels.append(f"{method} total")
    sheet = make_contact_sheet(images, labels, tonemap, exposure)
    if sheet is not None:
        iio.imwrite(os.path.join(output_dir, f"comparison_sheet_{view_name}.png"), sheet)


def eval_heldout_sh(args):
    device = torch.device(args.device)
    probe_specs = [
        ("latent", args.latent_probe_path),
        ("material", args.material_probe_path),
        ("latent_material", args.latent_material_probe_path),
    ]
    if args.residual_probe_path:
        probe_specs.append(("material_residual_latent", args.residual_probe_path))

    target_ref = torch.load(args.target_path, map_location="cpu")
    num_triangles = int(target_ref["valid_mask"].shape[0])
    fallback_direct = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
    fallback_indirect = torch.zeros((num_triangles, 4, 3), dtype=torch.float32)
    material_base_direct = None
    material_base_indirect = None
    if args.unfitted_fallback == "train_sh_mean":
        fallback_direct = target_ref["direct_sh_l1"].mean(dim=0, keepdim=True).repeat(num_triangles, 1, 1)
        fallback_indirect = target_ref["indirect_sh_l1"].mean(dim=0, keepdim=True).repeat(num_triangles, 1, 1)
    elif args.unfitted_fallback == "material":
        fallback_direct, fallback_indirect = predict_sh_coeffs_all(
            args.feature_cache_path,
            args.material_probe_path,
            device,
        )
        fallback_direct = clamp_coeffs_to_target_range(fallback_direct, target_ref["direct_sh_l1"], args.fallback_clip_pad)
        fallback_indirect = clamp_coeffs_to_target_range(fallback_indirect, target_ref["indirect_sh_l1"], args.fallback_clip_pad)
        material_base_direct = fallback_direct
        material_base_indirect = fallback_indirect

    if args.residual_probe_path and material_base_direct is None:
        material_base_direct, material_base_indirect = predict_sh_coeffs_all(
            args.feature_cache_path,
            args.material_probe_path,
            device,
        )
        material_base_direct = clamp_coeffs_to_target_range(material_base_direct, target_ref["direct_sh_l1"], args.fallback_clip_pad)
        material_base_indirect = clamp_coeffs_to_target_range(material_base_indirect, target_ref["indirect_sh_l1"], args.fallback_clip_pad)

    coeffs = {}
    sample_ref = None
    for method, path in probe_specs:
        target, sample, tri_ids, pred_direct, pred_indirect = predict_sh_coeffs(
            args.target_path,
            args.feature_cache_path,
            path,
            device,
        )
        if method == "material_residual_latent":
            full_direct = fallback_direct[:num_triangles].float().clone()
            full_indirect = fallback_indirect[:num_triangles].float().clone()
            full_direct[tri_ids] = material_base_direct[tri_ids].float() + pred_direct.float()
            full_indirect[tri_ids] = material_base_indirect[tri_ids].float() + pred_indirect.float()
        else:
            full_direct, full_indirect = sh_full_coeffs(
                target,
                tri_ids,
                pred_direct,
                pred_indirect,
                fallback_direct=fallback_direct,
                fallback_indirect=fallback_indirect,
            )
        coeffs[method] = {"direct": full_direct, "indirect": full_indirect}
        sample_ref = sample

    if args.mean_coeffs:
        target = target_ref
        direct_mean = target["direct_sh_l1"].mean(dim=0, keepdim=True)
        indirect_mean = target["indirect_sh_l1"].mean(dim=0, keepdim=True)
        coeffs["train_sh_mean"] = {
            "direct": direct_mean.repeat(num_triangles, 1, 1),
            "indirect": indirect_mean.repeat(num_triangles, 1, 1),
        }

    triangle_centers = sample_ref["triangle_centers"].float()
    gt_paths = sorted(glob.glob(args.gt_glob))
    if not gt_paths:
        raise FileNotFoundError(f"No held-out GT matched: {args.gt_glob}")
    os.makedirs(args.output_dir, exist_ok=True)

    metric_rows = []
    for gt_path in gt_paths:
        gt = read_gt_bundle(gt_path)
        view_name = os.path.splitext(os.path.basename(gt_path))[0]
        valid_pixel_mask = gt.get("valid_pixel_mask", gt["triangle_id_buffer"] >= 0).astype(bool)
        gt_direct = gt["I_direct"].astype(np.float32)
        gt_indirect = gt["I_indirect"].astype(np.float32)
        gt_total = gt_direct + gt_indirect
        predictions = {}
        view_dir = os.path.join(args.output_dir, view_name)
        os.makedirs(view_dir, exist_ok=True)
        tri_ids_img = gt["triangle_id_buffer"].astype(np.int64)
        valid_tri_pixels = (tri_ids_img >= 0) & (tri_ids_img < num_triangles)
        fitted_pixels = valid_tri_pixels & target_ref["valid_mask"].bool().cpu().numpy()[tri_ids_img.clip(0, num_triangles - 1)]
        unfitted_pixels = valid_pixel_mask & ~fitted_pixels
        fitted_mask = gather_fit_mask_image(target_ref["valid_mask"].bool(), gt)
        obs_count_image = gather_scalar_image(target_ref["obs_counts"].float(), gt)
        save_image_outputs(os.path.join(view_dir, "fitted_mask"), fitted_mask, "clip", 1.0, False)
        save_image_outputs(os.path.join(view_dir, "obs_count"), obs_count_image, "percentile", 1.0, False)

        for method, method_coeffs in coeffs.items():
            pred_direct = gather_sh_image(method_coeffs["direct"], gt, triangle_centers)
            pred_indirect = gather_sh_image(method_coeffs["indirect"], gt, triangle_centers)
            pred_total = pred_direct + pred_indirect
            predictions[method] = {"direct": pred_direct, "indirect": pred_indirect}
            save_image_outputs(os.path.join(view_dir, f"pred_direct_{method}"), pred_direct, args.tonemap, args.exposure, args.save_exr)
            save_image_outputs(os.path.join(view_dir, f"pred_indirect_{method}"), pred_indirect, args.tonemap, args.exposure, args.save_exr)
            save_image_outputs(os.path.join(view_dir, f"pred_total_{method}"), pred_total, args.tonemap, args.exposure, args.save_exr)
            save_diff_image_outputs(os.path.join(view_dir, f"diff5_direct_{method}"), gt_direct - pred_direct, args.diff_scale, args.save_exr)
            save_diff_image_outputs(os.path.join(view_dir, f"diff5_indirect_{method}"), gt_indirect - pred_indirect, args.diff_scale, args.save_exr)
            save_diff_image_outputs(os.path.join(view_dir, f"diff5_total_{method}"), gt_total - pred_total, args.diff_scale, args.save_exr)
            if args.debug_coeff_images:
                for comp_idx, comp_name in enumerate(["c0", "cx", "cy", "cz"]):
                    comp_img = gather_coeff_component_image(method_coeffs["indirect"], comp_idx, gt)
                    save_image_outputs(
                        os.path.join(view_dir, f"coeff_indirect_{comp_name}_{method}"),
                        comp_img,
                        "percentile",
                        1.0,
                        args.save_exr,
                    )
            neg_direct_img = gather_sh_negative_mask(method_coeffs["direct"], gt, triangle_centers)
            neg_indirect_img = gather_sh_negative_mask(method_coeffs["indirect"], gt, triangle_centers)
            neg_total_img = gather_sh_negative_mask(method_coeffs["direct"] + method_coeffs["indirect"], gt, triangle_centers)
            save_image_outputs(os.path.join(view_dir, f"negative_direct_mask_{method}"), neg_direct_img, "clip", 1.0, False)
            save_image_outputs(os.path.join(view_dir, f"negative_indirect_mask_{method}"), neg_indirect_img, "clip", 1.0, False)
            save_image_outputs(os.path.join(view_dir, f"negative_total_mask_{method}"), neg_total_img, "clip", 1.0, False)
            leak = component_leakage_metrics(pred_direct, pred_indirect, gt_direct, gt_indirect, valid_pixel_mask)
            all_mask = torch.from_numpy(valid_pixel_mask)
            fitted_mask_t = torch.from_numpy(fitted_pixels)
            unfitted_mask_t = torch.from_numpy(unfitted_pixels)

            def masked_psnr_np(pred_img, gt_img, mask):
                if int(mask.sum()) == 0:
                    return float("nan")
                return float(psnr(torch.from_numpy(pred_img)[mask], torch.from_numpy(gt_img)[mask]))

            metric_rows.append(
                {
                    "view": view_name,
                    "method": method,
                    "fitted_pixel_ratio": float(fitted_pixels.sum() / max(1, valid_pixel_mask.sum())),
                    "heldout_direct_psnr_all": masked_psnr_np(pred_direct, gt_direct, all_mask),
                    "heldout_indirect_psnr_all": masked_psnr_np(pred_indirect, gt_indirect, all_mask),
                    "heldout_total_psnr_all": masked_psnr_np(pred_total, gt_total, all_mask),
                    "heldout_direct_psnr_fitted_pixels_only": masked_psnr_np(pred_direct, gt_direct, fitted_mask_t),
                    "heldout_indirect_psnr_fitted_pixels_only": masked_psnr_np(pred_indirect, gt_indirect, fitted_mask_t),
                    "heldout_total_psnr_fitted_pixels_only": masked_psnr_np(pred_total, gt_total, fitted_mask_t),
                    "heldout_direct_psnr_unfitted_pixels_only": masked_psnr_np(pred_direct, gt_direct, unfitted_mask_t),
                    "heldout_indirect_psnr_unfitted_pixels_only": masked_psnr_np(pred_indirect, gt_indirect, unfitted_mask_t),
                    "heldout_total_psnr_unfitted_pixels_only": masked_psnr_np(pred_total, gt_total, unfitted_mask_t),
                    "direct_leakage_margin": leak["direct_leakage_margin_db"],
                    "indirect_leakage_margin": leak["indirect_leakage_margin_db"],
                }
            )

        save_image_outputs(os.path.join(view_dir, "gt_direct"), gt_direct, args.tonemap, args.exposure, args.save_exr)
        save_image_outputs(os.path.join(view_dir, "gt_indirect"), gt_indirect, args.tonemap, args.exposure, args.save_exr)
        save_image_outputs(os.path.join(view_dir, "gt_total"), gt_total, args.tonemap, args.exposure, args.save_exr)
        save_heldout_sheet(args.output_dir, view_name, gt_direct, gt_indirect, predictions, args.tonemap, args.exposure)

    metric_keys = [key for key in metric_rows[0].keys()]
    metrics_path = os.path.join(args.output_dir, "heldout_view_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write(",".join(metric_keys) + "\n")
        for row in metric_rows:
            f.write(",".join(str(row[key]) for key in metric_keys) + "\n")
    print(f"saved {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 0 RenderFormer latent radiance probing")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-latents", help="Save frozen view-independent triangle tokens")
    extract.add_argument("--h5_glob", required=True)
    extract.add_argument("--gt_h5_glob", default=None, help="Optional Stage-0 GT H5 glob. Defaults to reading GT fields from each input H5.")
    extract.add_argument("--output_dir", default="output/stage0_latents")
    extract.add_argument("--model_id", default="microsoft/renderformer-v1.1-swin-large")
    extract.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    extract.add_argument("--precision", choices=["float16", "bfloat16", "float32"], default="float16")
    extract.add_argument("--triangle_id_field", default="triangle_id_buffer")
    extract.add_argument("--direct_field", default="I_direct")
    extract.add_argument("--indirect_field", default="I_indirect")
    extract.add_argument("--total_field", default="I_total")
    extract.add_argument("--min_pixels", type=int, default=8)
    extract.set_defaults(func=extract_latents)

    train = sub.add_parser("train-probe", help="Train the direct/indirect MLP probe")
    train.add_argument("--cache_glob", default="output/stage0_latents/*.pt")
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train.add_argument("--batch_size", type=int, default=4096)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--hidden_dim", type=int, default=512)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--lambda_sum", type=float, default=0.5)
    train.add_argument("--output_model", default="output/stage0_probe/probe.pt")
    train.add_argument("--feature_key", choices=FEATURE_KEYS, default="latents")
    train.set_defaults(func=train_probe)

    eval_parser = sub.add_parser("eval-gather", help="Gather triangle probe predictions into images with a triangle ID buffer")
    eval_parser.add_argument("--cache_path", required=True, help="Latent cache .pt file from extract-latents")
    eval_parser.add_argument("--probe_path", required=True, help="Trained probe checkpoint")
    eval_parser.add_argument("--gt_path", required=True, help="Stage-0 GT .npz or .h5 containing triangle_id_buffer and GT passes")
    eval_parser.add_argument("--output_dir", default="output/stage0_eval")
    eval_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    eval_parser.add_argument("--hidden_dim", type=int, default=512)
    eval_parser.add_argument("--feature_key", choices=FEATURE_KEYS, default="latents")
    eval_parser.add_argument("--tonemap", choices=["reinhard", "clip", "percentile"], default="reinhard")
    eval_parser.add_argument("--exposure", type=float, default=1.0)
    eval_parser.add_argument("--save_exr", action="store_true", help="Also try to save float EXR files")
    eval_parser.set_defaults(func=eval_gather)

    compare = sub.add_parser("eval-compare-baselines", help="Compare latent probe, material-only probe, and triangle-mean baseline")
    compare.add_argument("--cache_path", required=True)
    compare.add_argument("--gt_path", required=True)
    compare.add_argument("--latent_probe_path", required=True)
    compare.add_argument("--material_probe_path", required=True)
    compare.add_argument("--latent_view_probe_path", default=None)
    compare.add_argument("--material_view_probe_path", default=None)
    compare.add_argument("--output_dir", default="output/stage0_compare")
    compare.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    compare.add_argument("--hidden_dim", type=int, default=512)
    compare.add_argument("--mean_cache_glob", default=None, help="Train cache glob for a fair train-only global mean baseline")
    compare.set_defaults(func=eval_compare)

    sh = sub.add_parser("fit-sh-l1", help="Fit per-triangle RGB SH L=1 coefficients from multi-view target caches")
    sh.add_argument("--cache_glob", required=True)
    sh.add_argument("--output_dir", default="output/stage0_sh_l1")
    sh.add_argument("--output_path", default=None)
    sh.add_argument("--min_views", type=int, default=4)
    sh.add_argument("--ridge", type=float, default=1e-4)
    sh.set_defaults(func=fit_sh_l1)

    sh_train = sub.add_parser("train-sh-probe", help="Train a probe that predicts direct/indirect SH L=1 coefficients")
    sh_train.add_argument("--target_path", required=True)
    sh_train.add_argument("--feature_cache_path", required=True)
    sh_train.add_argument("--feature_key", choices=FEATURE_KEYS + ["latent_material_features"], required=True)
    sh_train.add_argument("--output_model", required=True)
    sh_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sh_train.add_argument("--batch_size", type=int, default=2048)
    sh_train.add_argument("--epochs", type=int, default=300)
    sh_train.add_argument("--hidden_dim", type=int, default=512)
    sh_train.add_argument("--lr", type=float, default=1e-4)
    sh_train.add_argument("--weight_decay", type=float, default=1e-4)
    sh_train.add_argument("--lambda_sum", type=float, default=0.5)
    sh_train.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    sh_train.add_argument("--log_every", type=int, default=25)
    sh_train.add_argument("--no_feature_norm", action="store_true", help="Disable train-set feature mean/std normalization")
    sh_train.set_defaults(func=train_sh_probe)

    sh_residual = sub.add_parser("train-sh-residual-probe", help="Train latent probe on GT SH residual over a material SH baseline")
    sh_residual.add_argument("--target_path", required=True)
    sh_residual.add_argument("--feature_cache_path", required=True)
    sh_residual.add_argument("--material_probe_path", required=True)
    sh_residual.add_argument("--feature_key", choices=FEATURE_KEYS + ["latent_material_features"], default="latents")
    sh_residual.add_argument("--output_model", required=True)
    sh_residual.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sh_residual.add_argument("--batch_size", type=int, default=2048)
    sh_residual.add_argument("--epochs", type=int, default=300)
    sh_residual.add_argument("--hidden_dim", type=int, default=512)
    sh_residual.add_argument("--lr", type=float, default=1e-4)
    sh_residual.add_argument("--weight_decay", type=float, default=1e-4)
    sh_residual.add_argument("--lambda_sum", type=float, default=0.5)
    sh_residual.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    sh_residual.add_argument("--log_every", type=int, default=25)
    sh_residual.add_argument("--no_feature_norm", action="store_true", help="Disable train-set feature mean/std normalization")
    sh_residual.set_defaults(func=train_sh_residual_probe)

    sh_eval = sub.add_parser("eval-sh-probes", help="Evaluate SH probes in coefficient space")
    sh_eval.add_argument("--target_path", required=True)
    sh_eval.add_argument("--feature_cache_path", required=True)
    sh_eval.add_argument("--latent_probe_path", required=True)
    sh_eval.add_argument("--material_probe_path", required=True)
    sh_eval.add_argument("--latent_material_probe_path", required=True)
    sh_eval.add_argument("--output_dir", default="output/stage0_sh_eval")
    sh_eval.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sh_eval.set_defaults(func=eval_sh_probes)

    heldout = sub.add_parser("eval-heldout-sh", help="Render held-out views from predicted SH L=1 coefficients")
    heldout.add_argument("--target_path", required=True)
    heldout.add_argument("--feature_cache_path", required=True)
    heldout.add_argument("--gt_glob", required=True)
    heldout.add_argument("--latent_probe_path", required=True)
    heldout.add_argument("--material_probe_path", required=True)
    heldout.add_argument("--latent_material_probe_path", required=True)
    heldout.add_argument("--residual_probe_path", default=None)
    heldout.add_argument("--output_dir", default="output/stage0_sh_heldout")
    heldout.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    heldout.add_argument("--tonemap", choices=["reinhard", "clip", "percentile"], default="reinhard")
    heldout.add_argument("--exposure", type=float, default=1.0)
    heldout.add_argument("--save_exr", action="store_true")
    heldout.add_argument("--mean_coeffs", action="store_true")
    heldout.add_argument("--unfitted_fallback", choices=["material", "train_sh_mean", "zero"], default="material")
    heldout.add_argument("--fallback_clip_pad", type=float, default=0.1, help="Clip material fallback coefficients to train target range plus this fractional pad")
    heldout.add_argument("--diff_scale", type=float, default=5.0)
    heldout.add_argument("--debug_masks", action="store_true", help="Save fitted triangle mask images per held-out view")
    heldout.add_argument("--debug_coeff_images", action="store_true", help="Save c0/cx/cy/cz indirect coefficient images and negative masks")
    heldout.set_defaults(func=eval_heldout_sh)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
