import argparse
import csv
import glob
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


METHODS = {
    "latent": "latent",
    "material": "material",
    "mean": "train_sh_mean",
}


def style_gamma(x):
    return np.sqrt(np.maximum(x, 0.0))


def style_exposure(x):
    return 1.0 - np.exp(-3.0 * np.maximum(x, 0.0))


def style_soft_toon(x):
    return 1.0 / (1.0 + np.exp(-20.0 * (x - 0.5)))


def style_binary_lighting(x):
    return (x > 0.5).astype(np.float32)


def style_four_level_toon(x):
    return (np.floor(np.maximum(x, 0.0) * 4.0) / 4.0).astype(np.float32)


def style_shadow_removal(x):
    return np.power(np.maximum(x, 0.0), 0.2).astype(np.float32)


def style_anime_direct_light(x):
    luminance = 0.2126 * x[..., 0:1] + 0.7152 * x[..., 1:2] + 0.0722 * x[..., 2:3]
    return np.repeat((luminance > 0.5).astype(np.float32), 3, axis=-1)


def style_hard_toon(x):
    x = np.maximum(x, 0.0)
    return np.select(
        [x < 0.25, x < 0.50, x < 0.75],
        [0.15, 0.45, 0.75],
        default=1.0,
    ).astype(np.float32)


STYLES = {
    "binary_lighting": style_binary_lighting,
    "four_level_toon": style_four_level_toon,
    "shadow_removal": style_shadow_removal,
    "anime_direct_light": style_anime_direct_light,
    "hard_toon": style_hard_toon,
}


def read_gt_npz(path):
    with np.load(path) as gt:
        return {
            "I_direct": np.asarray(gt["I_direct"], dtype=np.float32),
            "I_indirect": np.asarray(gt["I_indirect"], dtype=np.float32),
            "I_total": np.asarray(gt["I_total"], dtype=np.float32)
            if "I_total" in gt
            else np.asarray(gt["I_direct"] + gt["I_indirect"], dtype=np.float32),
            "valid_pixel_mask": np.asarray(gt["valid_pixel_mask"], dtype=bool)
            if "valid_pixel_mask" in gt
            else np.ones(gt["I_direct"].shape[:2], dtype=bool),
        }


def read_exr(path):
    return np.asarray(iio.imread(path), dtype=np.float32)


def ldr_tonemap(image, mode="reinhard", exposure=1.0):
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    image = np.maximum(image * exposure, 0.0)
    if mode == "clip":
        mapped = np.clip(image, 0.0, 1.0)
    elif mode == "percentile":
        denom = np.percentile(image[image > 0], 99.0) if np.any(image > 0) else 1.0
        mapped = np.clip(image / max(float(denom), 1e-6), 0.0, 1.0)
    elif mode == "reinhard":
        mapped = image / (1.0 + image)
    else:
        raise ValueError(f"Unsupported tonemap: {mode}")
    return (mapped * 255.0 + 0.5).astype(np.uint8)


def save_png(path, image, tonemap="reinhard", exposure=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    iio.imwrite(path, ldr_tonemap(image, tonemap, exposure))


def save_sheet(path, images, labels, tonemap="reinhard", exposure=1.0):
    from PIL import Image, ImageDraw

    rendered = [ldr_tonemap(img, tonemap, exposure) for img in images]
    h, w = rendered[0].shape[:2]
    label_h = 20
    sheet = Image.new("RGB", (w * len(rendered), h + label_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for idx, (image, label) in enumerate(zip(rendered, labels)):
        sheet.paste(Image.fromarray(image), (idx * w, label_h))
        draw.text((idx * w + 4, 4), label, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    iio.imwrite(path, np.asarray(sheet))


def psnr(pred, target, mask):
    pred_t = torch.from_numpy(pred[mask])
    target_t = torch.from_numpy(target[mask])
    mse = F.mse_loss(pred_t, target_t).clamp_min(1e-8)
    max_val = target_t.max().clamp_min(1.0)
    return float(20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse))


def masked_mean_abs(a, b, mask):
    return float(np.mean(np.abs(a[mask] - b[mask])))


def ssim(pred, target, mask):
    valid = np.where(mask)
    if valid[0].size == 0:
        return 0.0
    y0, y1 = int(valid[0].min()), int(valid[0].max()) + 1
    x0, x1 = int(valid[1].min()), int(valid[1].max()) + 1
    pred_crop = pred[y0:y1, x0:x1].copy()
    target_crop = target[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]
    pred_crop[~mask_crop] = 0.0
    target_crop[~mask_crop] = 0.0

    x = torch.from_numpy(pred_crop).permute(2, 0, 1)[None].float()
    y = torch.from_numpy(target_crop).permute(2, 0, 1)[None].float()
    max_val = max(float(y.max()), 1.0)
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    kernel = 11
    padding = kernel // 2
    mu_x = F.avg_pool2d(x, kernel, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel, stride=1, padding=padding) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    score_image = score[0].permute(1, 2, 0)
    return float(score_image[torch.from_numpy(mask_crop).bool()].mean())


def main():
    parser = argparse.ArgumentParser(description="Stage 0 transport-aware stylization sanity check")
    parser.add_argument("--gt_glob", default="output/stage0_cbox_30views/stage0_gt/view_002*.npz")
    parser.add_argument("--prediction_root", default="output/stage0_cbox_30views/heldout_renderings_norm_debug")
    parser.add_argument("--output_dir", default="output/stage0_stylization")
    parser.add_argument("--tonemap", choices=["reinhard", "clip", "percentile"], default="reinhard")
    args = parser.parse_args()

    gt_paths = sorted(glob.glob(args.gt_glob))
    if not gt_paths:
        raise FileNotFoundError(f"No GT matched {args.gt_glob}")

    all_rows = []
    for style_name, style_fn in STYLES.items():
        style_dir = os.path.join(args.output_dir, style_name)
        rows = []
        for gt_path in gt_paths:
            view_name = os.path.splitext(os.path.basename(gt_path))[0]
            gt = read_gt_npz(gt_path)
            mask = gt["valid_pixel_mask"].astype(bool)
            gt_total = gt["I_total"]
            gt_post = style_fn(gt_total)
            gt_transport = style_fn(gt["I_direct"]) + gt["I_indirect"]
            direct_delta = float(np.mean(np.abs(style_fn(gt["I_direct"])[mask] - gt["I_direct"][mask])))
            effect_size = masked_mean_abs(gt_post, gt_transport, mask)

            view_pred_dir = os.path.join(args.prediction_root, view_name)
            fitted_mask_path = os.path.join(view_pred_dir, "fitted_mask.png")
            fitted_mask = (
                np.asarray(iio.imread(fitted_mask_path), dtype=np.float32)[..., :3] / 255.0
                if os.path.exists(fitted_mask_path)
                else np.repeat(mask[..., None].astype(np.float32), 3, axis=-1)
            )

            recon = {}
            for out_method, file_method in METHODS.items():
                pred_direct = read_exr(os.path.join(view_pred_dir, f"pred_direct_{file_method}.exr"))
                pred_indirect = read_exr(os.path.join(view_pred_dir, f"pred_indirect_{file_method}.exr"))
                recon[out_method] = style_fn(pred_direct) + pred_indirect

            save_png(os.path.join(style_dir, f"gt_post_{view_name}.png"), gt_post, args.tonemap)
            save_png(os.path.join(style_dir, f"gt_transport_{view_name}.png"), gt_transport, args.tonemap)
            for method, image in recon.items():
                save_png(os.path.join(style_dir, f"{method}_transport_{view_name}.png"), image, args.tonemap)
                save_png(
                    os.path.join(style_dir, f"diff_{method}_{view_name}.png"),
                    np.abs(image - gt_transport),
                    "percentile",
                )

            sheet_images = [
                gt["I_direct"],
                gt["I_indirect"],
                gt_total,
                gt_post,
                gt_transport,
                recon["latent"],
                recon["material"],
                recon["mean"],
                fitted_mask,
            ]
            sheet_labels = [
                "GT Direct",
                "GT Indirect",
                "GT Total",
                f"{style_name} Post",
                f"{style_name} Transport",
                "Latent Transport",
                "Material Transport",
                "Mean Transport",
                "Fitted Mask",
            ]
            save_sheet(os.path.join(style_dir, f"comparison_{view_name}.png"), sheet_images, sheet_labels, args.tonemap)

            post_vs_transport = {
                "style": style_name,
                "view": view_name,
                "kind": "gt_post_vs_transport",
                "psnr": psnr(gt_post, gt_transport, mask),
                "ssim": ssim(gt_post, gt_transport, mask),
                "mean_abs_diff": effect_size,
                "direct_delta": direct_delta,
                "transport_effect_size": effect_size,
            }
            rows.append(post_vs_transport)
            all_rows.append(post_vs_transport)

            for method, image in recon.items():
                row = {
                    "style": style_name,
                    "view": view_name,
                    "kind": f"{method}_transport",
                    "psnr": psnr(image, gt_transport, mask),
                    "ssim": ssim(image, gt_transport, mask),
                    "mean_abs_diff": masked_mean_abs(image, gt_transport, mask),
                    "direct_delta": direct_delta,
                    "transport_effect_size": effect_size,
                }
                rows.append(row)
                all_rows.append(row)

        metric_path = os.path.join(style_dir, "metrics.csv")
        os.makedirs(style_dir, exist_ok=True)
        with open(metric_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = []
    for style_name in STYLES:
        style_rows = [row for row in all_rows if row["style"] == style_name]
        for kind in ["gt_post_vs_transport", "latent_transport", "material_transport", "mean_transport"]:
            subset = [row for row in style_rows if row["kind"] == kind]
            summary.append(
                {
                    "style": style_name,
                    "kind": kind,
                    "psnr": float(np.mean([row["psnr"] for row in subset])),
                    "ssim": float(np.mean([row["ssim"] for row in subset])),
                    "mean_abs_diff": float(np.mean([row["mean_abs_diff"] for row in subset])),
                    "direct_delta": float(np.mean([row["direct_delta"] for row in subset])),
                    "transport_effect_size": float(np.mean([row["transport_effect_size"] for row in subset])),
                }
            )

    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"saved {summary_path}")
    for row in summary:
        print(
            f"{row['style']} {row['kind']} "
            f"psnr={row['psnr']:.4f} ssim={row['ssim']:.4f} "
            f"mean_abs_diff={row['mean_abs_diff']:.6f}"
        )


if __name__ == "__main__":
    main()
