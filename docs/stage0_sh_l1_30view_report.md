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

## v0.6 Debug Pass - Feature Normalization and Mask Images

### Code Changes

`experiments/stage0_probe.py` now normalizes SH probe input features during `train-sh-probe`.

```text
x_norm = (x - train_feature_mean) / train_feature_std
```

The checkpoint stores:

- `feature_mean`
- `feature_std`
- `feature_normalized`
- `h5_path`

`predict_sh_coeffs` and `eval-heldout-sh` apply the saved normalization before the MLP forward pass. They also validate that `target_path`, `feature_cache_path`, and checkpoint `h5_path` point to the same H5.

New heldout debug outputs:

| Output | Meaning |
| --- | --- |
| `fitted_mask.png` | White where `valid_mask[triangle_id_buffer[p]]` is true |
| `coeff_indirect_c0_<method>.png` | Indirect SH DC coefficient image |
| `coeff_indirect_cx_<method>.png` | Indirect SH x coefficient image |
| `coeff_indirect_cy_<method>.png` | Indirect SH y coefficient image |
| `coeff_indirect_cz_<method>.png` | Indirect SH z coefficient image |
| `negative_indirect_mask_<method>.png` | White where raw SH indirect value is negative before clamp |

Additional diagnostic script:

```text
experiments/debug_stage0_alignment.py
```

### Alignment Check

Command result:

```text
h5_triangles=5633 mapping_triangles=5633
object,offset,faces,mapping_count,max_abs_h5_vs_split_obj
background_0,0,128,128,0
background_1,128,128,128,0
background_2,256,128,128,0
background_3,384,128,128,0
tall_box,512,2560,2560,0
short_box,3072,2560,2560,0
light_0,5632,1,1,0
offset_total=5633
triangle_order_matches_split_objs=True
target_h5=output/stage0_cbox_30views/cbox_input.h5
feature_cache_h5=output/stage0_cbox_30views/cbox_input.h5
target_triangles=5633
feature_triangles=5633
fitted_triangles=396
```

This rules out the current suspected failure mode where cbox SH targets and feature cache come from different H5 files or different triangle ordering.

### Mask Coverage

For heldout `view_0020`:

```text
visible_pixels=15088
fitted_pixels=9990
fitted_pixel_ratio=0.662116
visible_unique_triangles=1951
fitted_visible_unique_triangles=379
```

Across heldout views `0020`-`0029`:

```text
fitted_pixel_ratio_mean=0.868132
fitted_pixel_ratio_min=0.662116
fitted_pixel_ratio_max=0.962027
visible_unique_triangles_mean=1132.5
fitted_visible_unique_triangles_mean=323.5
```

The black/empty regions are strongly correlated with triangles missing from the SH fitting set. The first-order cause is insufficient fitted triangle coverage, not scene/H5 mismatch.

### SH Negative Clamp Check

Mean negative indirect pixel ratio from debug masks:

| Method | Mean ratio | Max ratio |
| --- | ---: | ---: |
| latent | 0.0970 | 0.3464 |
| material-only | 0.0903 | 0.2796 |
| latent+material | 0.1120 | 0.3381 |

SH negativity exists and contributes locally, but it does not explain most of the large black side regions in `view_0020`. The fitted mask is the stronger signal.

### Normalized Probe Metrics

Coefficient-space, lower is better:

| Method | direct SH MSE | indirect SH MSE | total SH MSE | direct dir MSE | indirect dir MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent norm | 0.00326 | 0.00047 | 0.00309 | 0.00340 | 0.00050 |
| material norm | 0.05971 | 0.00400 | 0.07238 | 0.05170 | 0.00359 |
| latent+material norm | 0.00336 | 0.00046 | 0.00326 | 0.00350 | 0.00047 |

Heldout image-space average:

| Method | direct PSNR | indirect PSNR | total PSNR |
| --- | ---: | ---: | ---: |
| latent norm | 17.04 | 22.85 | 16.82 |
| material norm | 17.15 | 22.59 | 16.92 |
| latent+material norm | 17.05 | 22.85 | 16.82 |
| train SH mean | 14.21 | 19.73 | 13.90 |

Feature normalization improves the learned probes, especially indirect coefficient MSE, but material-only still has a small heldout total PSNR edge in this cbox setup.

## v0.7 Debug Pass - Interleaved Split and Fallback Metrics

### Changes

`eval-heldout-sh` now supports unfitted triangle fallback:

```text
--unfitted_fallback material
--unfitted_fallback train_sh_mean
--unfitted_fallback zero
```

Default is material fallback. In practice, material probe extrapolation on unfitted triangles can become extremely large, so material fallback coefficients are clipped to the train SH target coefficient range plus `--fallback_clip_pad`.

Heldout metrics are split into:

- `heldout_total_psnr_all`
- `heldout_total_psnr_fitted_pixels_only`
- `heldout_total_psnr_unfitted_pixels_only`

Every heldout view now saves:

- `fitted_mask.png`
- `obs_count.png`
- `negative_direct_mask_<method>.png`
- `negative_indirect_mask_<method>.png`
- `negative_total_mask_<method>.png`
- `diff5_direct_<method>.png`
- `diff5_indirect_<method>.png`
- `diff5_total_<method>.png`

`diff5_*` PNGs visualize `(GT - pred) * 5` with a signed midpoint, while EXR files store the raw difference.

### Interleaved Split

```text
test views  = 0,3,6,9,12,15,18,21,24,27
train views = 1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29
```

Artifacts:

| Path | Role |
| --- | --- |
| `output/stage0_cbox_30views_interleaved/cache_train/` | train cache |
| `output/stage0_cbox_30views_interleaved/cache_test/` | test cache |
| `output/stage0_cbox_30views_interleaved/stage0_gt_test/` | interleaved heldout GT |
| `output/stage0_cbox_30views_interleaved/sh_l1_targets.pt` | train-only SH target |
| `output/stage0_cbox_30views_interleaved/heldout_renderings_material_fallback_clipped_debug/` | material fallback eval |
| `output/stage0_cbox_30views_interleaved/heldout_renderings_trainmean_fallback_debug/` | train-mean fallback eval |

SH fitting:

| Train views | Fitted triangles | Total triangles |
| ---: | ---: | ---: |
| 20 | 407 | 5633 |

### Material Residual Latent Probe

New command:

```text
train-sh-residual-probe
```

This trains:

```text
material_pred = D_m(m_i)
residual_pred = D_r(h_i)
final_pred = material_pred + residual_pred
```

The residual target is:

```text
GT_SH - material_pred_SH
```

### Material Fallback Result

Material fallback uses clipped material coefficients for unfitted triangles.

| Method | total PSNR all | total PSNR fitted only | total PSNR unfitted only |
| --- | ---: | ---: | ---: |
| latent | 15.02 | 21.38 | 3.74 |
| material-only | 15.08 | 20.70 | 3.74 |
| latent+material | 15.02 | 21.38 | 3.74 |
| material residual latent | 15.01 | 21.38 | 3.74 |
| train SH mean | 12.10 | 13.05 | 9.77 |

Fitted pixels only: latent, latent+material, and residual latent are about `0.68 dB` better than material-only.

Unfitted pixels only: clipped material fallback is still poor. It avoids numeric explosion but is worse than train SH mean on unfitted regions.

### Train Mean Fallback Result

| Method | total PSNR all | total PSNR fitted only | total PSNR unfitted only |
| --- | ---: | ---: | ---: |
| latent | 17.85 | 21.38 | 9.77 |
| material-only | 17.91 | 20.70 | 9.77 |
| latent+material | 17.86 | 21.38 | 9.77 |
| material residual latent | 17.84 | 21.38 | 9.77 |
| train SH mean | 12.10 | 13.05 | 9.77 |

This confirms that all-pixel PSNR is dominated by fallback quality. Fitted-only PSNR is the cleaner signal for judging whether the learned SH probe is useful.

### Interpretation

- Fitted-only evaluation shows the latent signal is useful.
- Material fallback is not automatically safe for unfitted triangles; all-triangle material extrapolation can explode without clipping.
- Even clipped material fallback underperforms train SH mean in unfitted regions.
- The next useful fix is to train a dedicated fallback model or increase SH coverage, rather than relying on a probe trained only on fitted triangles to extrapolate to all unfitted geometry.
