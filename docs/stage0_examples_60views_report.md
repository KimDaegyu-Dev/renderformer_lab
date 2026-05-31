# Stage 0 Examples 60-View Coverage Report

## Setup

Scenes:

```text
examples/*.json, 15 scenes
```

Camera schedule:

```text
elevation = 10, 20, 30, 40 degrees
azimuths = 15 per elevation
views = 60 per scene
```

Train/test split:

```text
test = every 3rd view
train = remaining 40 views
```

Rendering:

```text
resolution = 128
spp = 64
flags = --stage0_gt --stage0_split_triangles --stage0_flip_camera_y
```

## Artifacts

| Artifact | Path |
| --- | --- |
| Manifest | `output/stage0_examples_60views/manifest_60views.json` |
| Coverage CSV | `output/stage0_examples_60views/sh_l1_coverage_summary.csv` |
| Scene GT | `output/stage0_examples_60views/<scene>/stage0_gt/view_*.npz` |
| Scene SH target | `output/stage0_examples_60views/<scene>/sh_l1_targets.pt` |
| Scene feature cache | `output/stage0_examples_60views/<scene>/feature_cache/*.pt` |

Scripts:

| Script | Purpose |
| --- | --- |
| `experiments/prepare_stage0_60views.py` | Generate 60-view configs |
| `experiments/fit_stage0_sh_l1_from_gt.py` | Fit SH L=1 targets directly from GT NPZ files |

## Coverage Summary

`min_views=6`, train views only.

| Scene | Fitted | Total | Ratio |
| --- | ---: | ---: | ---: |
| cbox-bunny | 478 | 6209 | 0.08 |
| cbox-lucy | 487 | 11803 | 0.04 |
| cbox-teapot | 471 | 9397 | 0.05 |
| cbox | 438 | 5633 | 0.08 |
| compose-scene | 251 | 7321 | 0.03 |
| constant-width | 260 | 4527 | 0.06 |
| crystals | 254 | 1949 | 0.13 |
| fox-in-the-wild | 296 | 1418 | 0.21 |
| horse-and-heart | 268 | 5023 | 0.05 |
| init-template | 506 | 513 | 0.99 |
| renderformer-logo | 438 | 6386 | 0.07 |
| room | 576 | 7141 | 0.08 |
| shader-ball | 515 | 11036 | 0.05 |
| tree | 384 | 4400 | 0.09 |
| veach-mis | 202 | 4575 | 0.04 |

## Notes

All 15 scenes rendered successfully with 60 GT views.

The first attempt to store full per-view feature caches ran out of disk space because the same scene latents were duplicated for every view. The final artifact layout avoids that:

- SH targets are fitted directly from GT NPZ files.
- Latent/material features are stored once per scene.

## Conclusion

Increasing from 30 to 60 orbit views is not enough to solve SH coverage for most object-heavy scenes. Most scenes still fit fewer than 10% of triangles at `min_views=6`.

The next useful direction is not just more orbit views, but better coverage sampling:

- cameras targeted at surface visibility,
- lower `min_views` for sparse regions,
- adaptive view generation from current `obs_counts`,
- or fallback models trained explicitly for unfitted triangles.
