# Stage 0 cbox 30-view SH L=1 Experiment

## Scope

This report records the v0.6 Stage 0 experiment:

- Render cbox with 30 orbit views.
- Use train views `0000`-`0019` only to fit SH L=1 targets.
- Hold out views `0020`-`0029` for image-space reconstruction.
- Compare latent, material-only, and latent+material SH probes.

## Artifacts

| Artifact | Path |
| --- | --- |
| 30-view config | `output/stage0_cbox_30views/cbox_30views.json` |
| 3-view sanity config | `output/stage0_cbox_30views/cbox_3view_sanity.json` |
| Stage 0 GT | `output/stage0_cbox_30views/stage0_gt/view_*.npz` |
| All caches | `output/stage0_cbox_30views/cache_all/` |
| Train caches | `output/stage0_cbox_30views/cache_train/` |
| Heldout caches | `output/stage0_cbox_30views/cache_test/` |
| Cache manifest | `output/stage0_cbox_30views/latents_multiview.pt` |
| SH L=1 targets | `output/stage0_cbox_30views/sh_l1_targets.pt` |
| Latent probe | `output/stage0_cbox_30views/probe_latent_to_sh_l1.pt` |
| Material probe | `output/stage0_cbox_30views/probe_material_to_sh_l1.pt` |
| Latent+material probe | `output/stage0_cbox_30views/probe_latent_material_to_sh_l1.pt` |
| Coefficient metrics | `output/stage0_cbox_30views/sh_l1_probe_metrics.csv` |
| Heldout metrics | `output/stage0_cbox_30views/heldout_view_metrics.csv` |
| Heldout images | `output/stage0_cbox_30views/heldout_renderings/` |

## SH Target Fitting

Settings:

```text
basis = [1, x, y, z]
min_views = 6
train_views = 0..19
heldout_views = 20..29
```

Result:

| Scene | View caches | Fitted triangles | Total triangles |
| --- | ---: | ---: | ---: |
| `stage0_cbox_30views` | 20 | 396 | 5633 |

The requested success target of at least 1000 valid SH triangles was not reached with `min_views=6`.

## Coefficient Metrics

Lower is better.

| Method | direct SH MSE | indirect SH MSE | total SH MSE | direct dir MSE | indirect dir MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent | 0.00496 | 0.00225 | 0.00377 | 0.00497 | 0.00219 |
| material-only | 0.06944 | 0.00551 | 0.08395 | 0.06175 | 0.00499 |
| latent+material | 0.00441 | 0.00194 | 0.00361 | 0.00452 | 0.00193 |

Latent and latent+material probes are clearly better than material-only in coefficient space.

## Heldout Image Metrics

Average over heldout views `0020`-`0029`. Higher PSNR is better.

| Method | direct PSNR | indirect PSNR | total PSNR | direct leakage margin | indirect leakage margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent | 16.89 | 21.91 | 16.66 | 5.05 | 8.19 |
| material-only | 16.69 | 22.22 | 16.76 | 3.16 | 8.05 |
| latent+material | 16.90 | 22.04 | 16.69 | 5.05 | 8.32 |
| train SH mean | 14.21 | 19.73 | 13.90 | -0.06 | 6.33 |

All learned probes beat train SH mean. Material-only is slightly ahead in heldout total PSNR, so this run does not establish heldout image-space superiority for the latent probe.

## Conclusion

The v0.6 pipeline works end to end:

- Camera Y flip is applied.
- 30 GT views were rendered.
- Multiview latent/material caches were created.
- SH L=1 targets were fitted from train views only.
- Three SH probes were trained.
- Heldout PNG/EXR reconstructions were exported through ID buffer gather.

The main blocker is coverage. With `min_views=6`, only 396 triangles are valid. The next experiment should increase usable multi-view triangle coverage before using heldout image-space metrics as the primary signal.
