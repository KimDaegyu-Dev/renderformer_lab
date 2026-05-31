# Stage 0 Experiment Artifact Versions

이 문서는 Stage 0 direct/indirect radiance probing 실험의 현재 산출물을 버전 단위로 기록한다.

## v0.1-cbox-single-view

상태: 완료

목적:

```text
Frozen RenderFormer view-independent per-triangle latent에서
direct / indirect RGB radiance를 작은 MLP probe로 디코딩할 수 있는지
cbox 단일 scene, 단일 view에서 sanity check한다.
```

### 코드 버전

변경 파일:

| 파일 | 변경 내용 |
| --- | --- |
| `renderformer/models/renderformer.py` | `extract_view_independent_latents(...)` hook 추가 |
| `experiments/stage0_probe.py` | latent 추출, GT target 로드, MLP probe 학습 CLI 추가 |
| `scene_processor/to_blend_stage0.py` | Stage 0 GT 렌더링 및 NPZ/H5 저장 경로 보강 |
| `docs/repo_architecture_stage0.md` | 레포 구조, Stage 0 설계, 진행 상태 문서화 |
| `.gitignore` | 임시 Blender dependency 폴더 `.blender_py/` ignore 추가 |

주의:

- `scene_processor/to_blend_stage0.py`는 사용자가 추가한 파일을 바탕으로, 현재 환경에서 Blender 내장 API fallback과 NPZ fallback을 추가해 실행했다.
- Blender Python에 `h5py`가 없어 GT는 `.h5`가 아니라 `.npz`로 저장됐다.

### 데이터/렌더 설정

| 항목 | 값 |
| --- | --- |
| Scene | `examples/cbox.json` |
| RenderFormer input | `output/stage0_cbox/cbox_input.h5` |
| GT output | `output/stage0_cbox/stage0_gt/view_0000.npz` |
| Resolution | `128 x 128` |
| Samples per pixel | `64` |
| GT renderer | Blender 4.4 / Cycles |
| GT passes | Combined, Diffuse Direct, Diffuse Indirect, Object Index |
| Triangle ID strategy | `--stage0_split_triangles`로 triangle별 Blender object 생성 후 Object Index 사용 |
| `min_pixels` | `8` |

### GT 산출물

디렉토리:

```text
output/stage0_cbox/stage0_gt/
```

파일:

| 파일 | 역할 |
| --- | --- |
| `triangle_id_mapping.json` | original object/polygon과 stage0 triangle_id 매핑 |
| `view_0000_combined_0001.exr` | Blender Combined pass |
| `view_0000_diffuse_direct_0001.exr` | Diffuse Direct pass |
| `view_0000_diffuse_indirect_0001.exr` | Diffuse Indirect pass |
| `view_0000_object_index_0001.exr` | Object Index pass |
| `view_0000.npz` | 학습용 Stage 0 GT bundle |

`view_0000.npz` 필드:

```text
I_total:            [128, 128, 3]
I_direct:           [128, 128, 3]
I_indirect:         [128, 128, 3]
triangle_id_buffer: [128, 128]
valid_pixel_mask:   [128, 128]
c2w:                [4, 4]
L_direct_tri:       [5633, 3]
L_indirect_tri:     [5633, 3]
L_total_tri:        [5633, 3]
tri_visible_count:  [5633]
tri_valid_mask:     [5633]
```

GT 통계:

```text
triangles: 5633
valid visible pixels: 13326
visible/supervised triangles: 394
max visible pixels per triangle: 55
triangle_id range: -1 to 5247
I_direct max: 4.029329776763916
I_indirect max: 1.6118742227554321
I_total max: 2.2667033672332764
```

### Latent 산출물

파일:

```text
output/stage0_latents_cbox/cbox_input.pt
```

내용:

```text
latents: [5633, 1024]
valid_mask: [5633]
direct target: [5633, 3]
indirect target: [5633, 3]
total target: [5633, 3]
target_mask: [5633]
supervised triangles: 394
direct target max: 4.028366565704346
indirect target max: 0.9387350082397461
```

생성 명령:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py extract-latents \
  --h5_glob "output/stage0_cbox/cbox_input.h5" \
  --gt_h5_glob "output/stage0_cbox/stage0_gt/view_0000.npz" \
  --output_dir output/stage0_latents_cbox \
  --device cuda \
  --precision float16
```

### Probe 학습 산출물

초기 run:

```text
checkpoint: output/stage0_probe_cbox/probe.pt
epochs: 100
batch_size: 512
lr: 1e-3
final direct_psnr: 9.01
final indirect_psnr: 10.39
final total_psnr: 8.67
```

관찰:

```text
lr=1e-3은 작은 단일-scene 데이터셋에서 초반 이후 정체했다.
```

최종 sanity-check run:

```text
checkpoint: output/stage0_probe_cbox/probe_lr1e-4.pt
epochs: 300
batch_size: 512
lr: 1e-4
lambda_sum: 0.5
final direct_psnr: 30.07
final indirect_psnr: 15.71
final total_psnr: 31.83
```

생성 명령:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py train-probe \
  --cache_glob "output/stage0_latents_cbox/*.pt" \
  --epochs 300 \
  --batch_size 512 \
  --lr 1e-4 \
  --output_model output/stage0_probe_cbox/probe_lr1e-4.pt \
  --device cuda
```

### 해석

이번 버전은 Stage 0 pipeline이 끝까지 연결되는지 확인한 첫 산출물이다.

성공한 점:

- cbox scene에서 triangle ID 기반 direct/indirect GT를 생성했다.
- RenderFormer input H5와 Stage 0 GT를 분리해 latent cache를 만들었다.
- frozen VI latent에서 direct/indirect target으로 MLP probe를 학습했다.
- 단일 scene overfit 기준으로 total reconstruction 지표가 30 dB 이상까지 올라갔다.

제한:

- 단일 scene, 단일 view라 일반화 성능을 말할 수 없다.
- Blender Diffuse Direct/Indirect pass 기준이라 specular transport decomposition은 아직 포함하지 않는다.
- `indirect_psnr`은 direct/total에 비해 낮아 component 분리 난이도가 남아 있다.
- 아직 ID buffer gather image reconstruction, leakage metric, material-only baseline이 없다.

## v0.2-cbox-image-reconstruction

상태: 완료

목표:

```text
probe prediction을 triangle_id_buffer로 gather해서
pred_direct_img, pred_indirect_img, pred_total_img를 저장하고
GT pass와 이미지 공간 PSNR/visual comparison을 만든다.
```

### 코드 변경

`experiments/stage0_probe.py`에 `eval-gather` subcommand를 추가했다.

기능:

- `probe_lr1e-4.pt`가 예측한 per-triangle direct/indirect RGB를 로드한다.
- `triangle_id_buffer`로 per-pixel predicted image를 gather한다.
- predicted direct, indirect, total 이미지를 저장한다.
- GT direct, indirect, combined total, direct+indirect recomposed total을 저장한다.
- absolute difference 이미지와 comparison sheet를 저장한다.
- visible pixel mask 기준 image-space PSNR을 기록한다.

### 실행 명령

```bash
conda run -n renderformer python -B experiments/stage0_probe.py eval-gather \
  --cache_path output/stage0_latents_cbox/cbox_input.pt \
  --probe_path output/stage0_probe_cbox/probe_lr1e-4.pt \
  --gt_path output/stage0_cbox/stage0_gt/view_0000.npz \
  --output_dir output/stage0_eval_cbox \
  --device cuda \
  --tonemap reinhard \
  --save_exr
```

### 산출물

디렉토리:

```text
output/stage0_eval_cbox/
```

대표 PNG:

| 파일 | 역할 |
| --- | --- |
| `comparison_sheet.png` | GT/pred direct, indirect, total 비교 시트 |
| `pred_direct.png` | predicted direct image |
| `pred_indirect.png` | predicted indirect image |
| `pred_total.png` | predicted direct + indirect image |
| `gt_direct.png` | GT Diffuse Direct pass |
| `gt_indirect.png` | GT Diffuse Indirect pass |
| `gt_recomposed.png` | GT direct + indirect |
| `gt_total.png` | Blender Combined pass |
| `absdiff_direct.png` | direct absolute error |
| `absdiff_indirect.png` | indirect absolute error |
| `absdiff_total_recomposed.png` | predicted total vs GT direct+indirect absolute error |

EXR도 같은 prefix로 저장했다. `metrics.txt`도 함께 저장된다.

### Image-Space Metrics

visible pixel mask 기준:

```text
direct_psnr_img: 16.29327964782715
indirect_psnr_img: 19.55465316772461
total_psnr_vs_combined_img: 7.722713947296143
total_psnr_vs_direct_plus_indirect_img: 17.379226684570312
valid_pixels: 13326
```

해석:

- Stage 0 목적상 `total_psnr_vs_direct_plus_indirect_img`가 더 직접적인 지표다.
- `Blender Combined` pass는 emission/combined shading 정의 차이 때문에 `Diffuse Direct + Diffuse Indirect`와 정확히 같지 않아 `total_psnr_vs_combined_img`가 낮다.
- visual comparison 기준으로 direct prediction은 box/room geometry를 따라가지만, 일부 색 leakage가 보인다.
- indirect prediction은 저주파 global illumination 형태는 잡지만 component 분리 품질은 아직 개선 여지가 있다.

### 다음 버전 후보

`v0.3-cbox-yflip-baselines`

목표:

```text
RenderFormer latent probe가 material-only baseline과 triangle mean baseline보다 나은지 비교한다.
```

## v0.3-cbox-yflip-baselines

상태: 완료

목표:

```text
cbox Stage 0 GT의 카메라 local Y/up 축을 반전해 다시 렌더링하고,
RenderFormer latent probe를 material-only baseline 및 triangle mean baseline과 비교한다.
```

### 코드 변경

| 파일 | 변경 내용 |
| --- | --- |
| `scene_processor/to_blend_stage0.py` | `--stage0_flip_camera_y` 옵션 추가. fallback camera 생성 시 local Z roll을 적용해 camera local Y/up 방향을 반전한다. |
| `experiments/stage0_probe.py` | material-only feature 생성, `--feature_key material_features`, `eval-compare-baselines` subcommand 추가 |
| `renderformer/models/renderformer.py` | Stage 0 latent extraction hook 유지 |
| `docs/repo_architecture_stage0.md` | y-flipped cbox 진행 상태와 baseline 비교 결과 반영 |
| `docs/stage0_experiment_versions.md` | v0.3 산출물 버전 기록 |

### 데이터/렌더 설정

| 항목 | 값 |
| --- | --- |
| Scene | `examples/cbox.json` |
| RenderFormer input | `output/stage0_cbox_yflip/cbox_input.h5` |
| GT output | `output/stage0_cbox_yflip/stage0_gt/view_0000.npz` |
| Resolution | `128 x 128` |
| Samples per pixel | `64` |
| GT renderer | Blender 4.4 / Cycles |
| GT passes | Combined, Diffuse Direct, Diffuse Indirect, Object Index |
| Camera fix | `--stage0_flip_camera_y` |
| Triangle ID strategy | `--stage0_split_triangles`로 triangle별 Blender object 생성 후 Object Index 사용 |
| `min_pixels` | `8` |

GT 통계:

```text
triangles: 5633
valid visible pixels: 13376
visible/supervised triangles: 397
max visible pixels per triangle: 55
triangle_id range: -1 to 5247
I_direct max: 4.028948783874512
I_indirect max: 1.4941611289978027
I_total max: 2.2094061374664307
```

### 실행 명령

RenderFormer input H5:

```bash
conda run -n renderformer python scene_processor/convert_scene.py examples/cbox.json \
  --mesh_path output/stage0_cbox_yflip/cbox.obj \
  --output_h5_path output/stage0_cbox_yflip/cbox_input.h5
```

GT 렌더링:

```bash
blender --background --factory-startup --python-expr "..."  # to_blend_stage0.py 실행
```

실제 Stage 0 인자:

```text
scene_config: examples/cbox.json
output_dir:   output/stage0_cbox_yflip
mesh_path:    output/stage0_cbox_yflip/cbox.obj
resolution:   128
spp:          64
flags:        --stage0_gt --stage0_split_triangles --stage0_flip_camera_y
```

Latent/material cache:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py extract-latents \
  --h5_glob "output/stage0_cbox_yflip/cbox_input.h5" \
  --gt_h5_glob "output/stage0_cbox_yflip/stage0_gt/view_0000.npz" \
  --output_dir output/stage0_latents_cbox_yflip \
  --device cuda \
  --precision float16
```

RenderFormer latent probe 학습:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py train-probe \
  --cache_glob "output/stage0_latents_cbox_yflip/*.pt" \
  --epochs 300 \
  --batch_size 512 \
  --lr 1e-4 \
  --output_model output/stage0_probe_cbox_yflip/probe_latent_lr1e-4.pt \
  --device cuda \
  --feature_key latents
```

Material-only baseline 학습:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py train-probe \
  --cache_glob "output/stage0_latents_cbox_yflip/*.pt" \
  --epochs 300 \
  --batch_size 512 \
  --lr 1e-4 \
  --output_model output/stage0_probe_cbox_yflip/probe_material_lr1e-4.pt \
  --device cuda \
  --feature_key material_features
```

Image reconstruction:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py eval-gather \
  --cache_path output/stage0_latents_cbox_yflip/cbox_input.pt \
  --probe_path output/stage0_probe_cbox_yflip/probe_latent_lr1e-4.pt \
  --gt_path output/stage0_cbox_yflip/stage0_gt/view_0000.npz \
  --output_dir output/stage0_eval_cbox_yflip_latent \
  --device cuda \
  --tonemap reinhard \
  --save_exr
```

Baseline comparison:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py eval-compare-baselines \
  --cache_path output/stage0_latents_cbox_yflip/cbox_input.pt \
  --gt_path output/stage0_cbox_yflip/stage0_gt/view_0000.npz \
  --latent_probe_path output/stage0_probe_cbox_yflip/probe_latent_lr1e-4.pt \
  --material_probe_path output/stage0_probe_cbox_yflip/probe_material_lr1e-4.pt \
  --output_dir output/stage0_compare_cbox_yflip \
  --device cuda
```

### 학습 결과

RenderFormer latent probe:

```text
checkpoint: output/stage0_probe_cbox_yflip/probe_latent_lr1e-4.pt
epochs: 300
batch_size: 512
lr: 1e-4
final direct_psnr: 39.52
final indirect_psnr: 37.23
final total_psnr: 40.43
```

Material-only baseline:

```text
checkpoint: output/stage0_probe_cbox_yflip/probe_material_lr1e-4.pt
epochs: 300
batch_size: 512
lr: 1e-4
final direct_psnr: 28.58
final indirect_psnr: 20.09
final total_psnr: 26.73
```

주의:

```text
위 학습 PSNR은 supervised triangle 평균 target 기준이다.
최종 비교는 triangle_id_buffer gather 이후 visible image pixel 기준 PSNR을 우선한다.
```

### 산출물

| 경로 | 역할 |
| --- | --- |
| `output/stage0_cbox_yflip/stage0_gt/view_0000.npz` | y-flipped Stage 0 GT bundle |
| `output/stage0_latents_cbox_yflip/cbox_input.pt` | latent/material feature cache |
| `output/stage0_probe_cbox_yflip/probe_latent_lr1e-4.pt` | RenderFormer latent probe checkpoint |
| `output/stage0_probe_cbox_yflip/probe_material_lr1e-4.pt` | material-only baseline checkpoint |
| `output/stage0_eval_cbox_yflip_latent/` | latent probe PNG/EXR reconstruction |
| `output/stage0_eval_cbox_yflip_material/` | material-only PNG/EXR reconstruction |
| `output/stage0_compare_cbox_yflip/baseline_comparison_metrics.csv` | baseline 비교 CSV |

`output/stage0_eval_cbox_yflip_latent/` 대표 PNG:

| 파일 | 역할 |
| --- | --- |
| `comparison_sheet.png` | GT/pred direct, indirect, total 비교 시트 |
| `pred_direct.png` | predicted direct image |
| `pred_indirect.png` | predicted indirect image |
| `pred_total.png` | predicted direct + indirect image |
| `gt_direct.png` | GT Diffuse Direct pass |
| `gt_indirect.png` | GT Diffuse Indirect pass |
| `gt_recomposed.png` | GT direct + indirect |
| `gt_total.png` | Blender Combined pass |
| `absdiff_direct.png` | direct absolute error |
| `absdiff_indirect.png` | indirect absolute error |
| `absdiff_total_recomposed.png` | predicted total vs GT direct+indirect absolute error |

EXR도 같은 prefix로 저장했다.

### Image-Space Metrics

visible pixel mask 기준:

| Method | direct PSNR | indirect PSNR | total PSNR vs direct+indirect |
| --- | ---: | ---: | ---: |
| RenderFormer latent probe | 15.7455 | 21.5936 | 16.7714 |
| Material-only baseline | 22.2190 | 21.4119 | 22.7824 |
| Triangle mean baseline | 10.7402 | 18.8634 | 12.1243 |

Latent probe의 추가 metric:

```text
total_psnr_vs_combined_img: 8.15962028503418
valid_pixels: 13376
```

Material-only baseline의 추가 metric:

```text
total_psnr_vs_combined_img: 7.0446858406066895
valid_pixels: 13376
```

### 해석

결론:

```text
현재 단일 cbox, 단일 view, train=test overfit 조건에서는
RenderFormer latent probe가 triangle mean baseline보다 낫지만,
material-only baseline보다 낫지는 않다.
```

세부 관찰:

- latent probe는 triangle mean baseline 대비 direct/indirect/total 모두 높다.
- material-only baseline은 direct와 total image-space PSNR에서 latent probe보다 높다.
- latent probe는 supervised triangle target 학습 PSNR은 높지만, image-space gather에서는 visible pixel 가중/triangle target 평균화 차이와 component leakage 영향이 더 크게 드러난다.
- 이 결과는 "latent가 material보다 표현력이 없다"는 결론이라기보다, 현재 단일 scene overfit 설정이 material/color prior에 매우 유리하다는 의미에 가깝다.

다음 버전 후보:

```text
v0.4-component-leakage-and-multiscene
```

목표:

```text
component leakage metric을 추가하고,
여러 scene/view split에서 material-only baseline 대비 latent probe의 일반화 성능을 재평가한다.
```

## v0.4-multiscene-generalization

상태: 완료

목표:

```text
component leakage metric을 추가하고,
cbox 외 단순 scene을 포함한 train/test split에서
RenderFormer latent probe와 material-only baseline의 일반화 성능을 비교한다.
```

### 코드 변경

| 파일 | 변경 내용 |
| --- | --- |
| `experiments/stage0_probe.py` | component leakage metric 추가, multi-scene cache 파일명 충돌 방지 |
| `scene_processor/to_blend_stage0.py` | vertex-color diffuse 연결 fallback 추가. `Group` 노드가 없으면 `Principled BSDF/Base Color`에 연결 |
| `docs/repo_architecture_stage0.md` | multi-scene v1 결과와 다음 작업 갱신 |
| `docs/stage0_experiment_versions.md` | v0.4 실험 산출물 기록 |

### Component Leakage Metric

정의:

```text
direct_cross_psnr_img       = PSNR(pred_direct, GT_indirect)
indirect_cross_psnr_img     = PSNR(pred_indirect, GT_direct)
direct_leakage_margin_db    = PSNR(pred_direct, GT_direct) - PSNR(pred_direct, GT_indirect)
indirect_leakage_margin_db  = PSNR(pred_indirect, GT_indirect) - PSNR(pred_indirect, GT_direct)
```

해석:

```text
leakage margin이 클수록 예측 component가 반대 component보다 자기 component에 더 가깝다.
margin이 0 이하이면 component identity가 불안정하거나 scene 평균에 가까운 예측일 수 있다.
```

### Dataset

디렉토리:

```text
output/stage0_multiscene_v1/
```

공통 설정:

| 항목 | 값 |
| --- | --- |
| Resolution | `128 x 128` |
| Samples per pixel | `64` |
| GT renderer | Blender 4.4 / Cycles |
| GT passes | Combined, Diffuse Direct, Diffuse Indirect, Object Index |
| Camera fix | `--stage0_flip_camera_y` |
| Triangle ID strategy | `--stage0_split_triangles` |

Scene split:

| Split | Scenes |
| --- | --- |
| train | `cbox`, `cbox-bunny`, `cbox-teapot`, `constant-width` |
| test | `compose-scene`, `shader-ball` |

생성된 GT bundle:

| Scene | GT |
| --- | --- |
| `cbox` | `output/stage0_multiscene_v1/cbox/stage0_gt/view_0000.npz` |
| `cbox-bunny` | `output/stage0_multiscene_v1/cbox-bunny/stage0_gt/view_0000.npz` |
| `cbox-teapot` | `output/stage0_multiscene_v1/cbox-teapot/stage0_gt/view_0000.npz` |
| `constant-width` | `output/stage0_multiscene_v1/constant-width/stage0_gt/view_0000.npz` |
| `compose-scene` | `output/stage0_multiscene_v1/compose-scene/stage0_gt/view_0000.npz` |
| `shader-ball` | `output/stage0_multiscene_v1/shader-ball/stage0_gt/view_0000.npz` |

### Latent Cache

디렉토리:

```text
output/stage0_latents_multiscene_v1/
```

생성 명령:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py extract-latents \
  --h5_glob "output/stage0_multiscene_v1/*/input.h5" \
  --gt_h5_glob "output/stage0_multiscene_v1/*/stage0_gt/view_0000.npz" \
  --output_dir output/stage0_latents_multiscene_v1 \
  --device cuda \
  --precision float16
```

split cache:

```text
train cache: output/stage0_latents_multiscene_v1_train/
test cache:  output/stage0_latents_multiscene_v1_test/
```

### Probe 학습

RenderFormer latent probe:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py train-probe \
  --cache_glob "output/stage0_latents_multiscene_v1_train/*.pt" \
  --epochs 300 \
  --batch_size 2048 \
  --lr 1e-4 \
  --output_model output/stage0_probe_multiscene_v1/probe_latent_lr1e-4.pt \
  --device cuda \
  --feature_key latents
```

최종 train target 지표:

```text
direct_psnr: 30.95
indirect_psnr: 16.78
total_psnr: 31.43
```

Material-only baseline:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py train-probe \
  --cache_glob "output/stage0_latents_multiscene_v1_train/*.pt" \
  --epochs 300 \
  --batch_size 2048 \
  --lr 1e-4 \
  --output_model output/stage0_probe_multiscene_v1/probe_material_lr1e-4.pt \
  --device cuda \
  --feature_key material_features
```

최종 train target 지표:

```text
direct_psnr: 27.47
indirect_psnr: 18.93
total_psnr: 25.91
```

### Evaluation Artifacts

| 경로 | 역할 |
| --- | --- |
| `output/stage0_compare_multiscene_v1/per_scene_metrics.csv` | scene별 latent/material/triangle mean 비교 |
| `output/stage0_compare_multiscene_v1/split_summary_metrics.csv` | train/test split 평균 비교 |
| `output/stage0_eval_multiscene_v1/compose-scene/latent/` | test scene latent PNG/EXR |
| `output/stage0_eval_multiscene_v1/compose-scene/material/` | test scene material-only PNG/EXR |
| `output/stage0_eval_multiscene_v1/shader-ball/latent/` | test scene latent PNG/EXR |
| `output/stage0_eval_multiscene_v1/shader-ball/material/` | test scene material-only PNG/EXR |

### Split Summary

visible pixel mask 기준 평균:

| Split | Method | direct PSNR | indirect PSNR | total PSNR | direct leakage margin | indirect leakage margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | RenderFormer latent | 11.73 | 21.96 | 14.35 | 7.70 | 13.43 |
| train | Material-only | 12.77 | 24.87 | 15.49 | 7.88 | 16.08 |
| train | Triangle mean | 12.03 | 26.48 | 15.15 | 1.69 | 17.73 |
| test | RenderFormer latent | 9.79 | 17.06 | 10.05 | 5.99 | 6.23 |
| test | Material-only | 9.30 | 21.00 | 10.01 | 7.25 | 10.08 |
| test | Triangle mean | 12.85 | 31.35 | 13.92 | -1.77 | 20.96 |

### Per-Test Scene Results

`compose-scene`:

| Method | direct PSNR | indirect PSNR | total PSNR | direct margin | indirect margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| RenderFormer latent | 5.31 | 12.23 | 3.64 | 5.29 | 0.57 |
| Material-only | 3.03 | 18.70 | 2.33 | 5.26 | 6.90 |
| Triangle mean | 12.45 | 38.41 | 12.41 | -4.00 | 28.01 |

`shader-ball`:

| Method | direct PSNR | indirect PSNR | total PSNR | direct margin | indirect margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| RenderFormer latent | 14.27 | 21.90 | 16.45 | 6.70 | 11.90 |
| Material-only | 15.58 | 23.30 | 17.68 | 9.25 | 13.26 |
| Triangle mean | 13.25 | 24.29 | 15.43 | 0.45 | 13.91 |

### 해석

결론:

```text
multi-scene v1의 test split에서는 RenderFormer latent probe가
material-only baseline보다 direct와 total에서 아주 근소하게 높지만,
indirect와 leakage margin은 material-only baseline이 더 좋다.
따라서 현재 결과는 latent의 명확한 일반화 우세를 보이지 않는다.
```

세부 관찰:

- test total PSNR 평균은 latent `10.0461`, material-only `10.0051`로 차이가 0.04 dB 수준이다.
- material-only는 test indirect PSNR에서 `21.00`으로 latent `17.06`보다 높다.
- latent는 `compose-scene` direct/total에서 material-only보다 낫지만, `shader-ball`에서는 material-only가 더 낫다.
- train target 기준 latent는 material-only보다 direct/total 학습 지표가 높지만, image-space generalization에서는 그 차이가 유지되지 않는다.
- 현재 triangle mean은 각 평가 scene의 GT target 평균을 쓰는 per-scene mean baseline이다. 일반화 baseline으로 쓰려면 train-set global mean으로 별도 분리해야 한다.

다음 버전 후보:

```text
v0.5-train-mean-baseline-and-more-views
```

목표:

```text
triangle mean baseline을 train-only mean으로 엄밀화하고,
scene/view 수를 늘려 material-only 대비 latent probe의 일반화 차이를 더 안정적으로 측정한다.
```

## v0.5-train-mean-viewdir-sh

상태: 완료

목표:

```text
per-scene triangle mean baseline의 test leakage를 제거하고,
D(h_i, n_i, v_i) decoder를 추가하며,
multi-view sample에서 SH L=1 target을 fitting하는 경로를 만든다.
```

### 코드 변경

| 파일 | 변경 내용 |
| --- | --- |
| `experiments/stage0_probe.py` | train-only global mean baseline 추가 |
| `experiments/stage0_probe.py` | `triangle_normals`, `view_dirs`, `latent_view_features`, `material_view_features` cache 저장 |
| `experiments/stage0_probe.py` | `latent_view_features` / `material_view_features` 학습 및 평가 지원 |
| `experiments/stage0_probe.py` | 단일 H5 + 여러 GT view cache 추출 지원 |
| `experiments/stage0_probe.py` | `fit-sh-l1` command 추가 |

### Train-Only Mean Baseline

이전 `triangle_mean` baseline은 평가 scene의 GT target 평균을 직접 사용했다. v0.5에서는 test scene 평균 사용을 금지하고 train split에서만 구한 global RGB mean을 쓴다.

정의:

```text
mean_direct   = pixel_count-weighted mean(L_direct_tri) over train supervised triangles
mean_indirect = pixel_count-weighted mean(L_indirect_tri) over train supervised triangles
```

평가 시 모든 triangle에 같은 train-global mean RGB를 할당한다.

### View-Aware Decoder

새 feature:

```text
latent_view_features = concat(h_i, n_i, v_i)
```

여기서:

```text
h_i: RenderFormer view-independent latent
n_i: averaged vertex normal
v_i: normalized(camera_position - triangle_center)
```

목표 형태:

```text
D(h_i, n_i, v_i) -> (L_direct_i, L_indirect_i)
```

### 실행 산출물

| 경로 | 역할 |
| --- | --- |
| `output/stage0_latents_multiscene_v2/` | view feature 포함 6-scene cache |
| `output/stage0_latents_multiscene_v2_train/` | train cache |
| `output/stage0_latents_multiscene_v2_test/` | test cache |
| `output/stage0_probe_multiscene_v2/probe_latent_view_lr1e-4.pt` | `D(h,n,v)` checkpoint |
| `output/stage0_compare_multiscene_v2/per_scene_metrics.csv` | scene별 비교 |
| `output/stage0_compare_multiscene_v2/split_summary_metrics.csv` | split 평균 비교 |

### Split Summary

visible pixel mask 기준 평균:

| Split | Method | direct PSNR | indirect PSNR | total PSNR | direct margin | indirect margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | RenderFormer latent | 11.73 | 21.96 | 14.35 | 7.70 | 13.43 |
| train | RenderFormer latent + view | 12.66 | 19.37 | 14.43 | 8.20 | 10.38 |
| train | Material-only | 12.77 | 24.87 | 15.49 | 7.88 | 16.08 |
| train | Train-global mean | 10.73 | 23.42 | 13.44 | 3.10 | 14.58 |
| test | RenderFormer latent | 9.79 | 17.06 | 10.05 | 5.99 | 6.23 |
| test | RenderFormer latent + view | 9.68 | 13.66 | 9.50 | 6.03 | 2.50 |
| test | Material-only | 9.30 | 21.00 | 10.01 | 7.25 | 10.08 |
| test | Train-global mean | 9.98 | 20.05 | 10.25 | 3.58 | 9.01 |

### SH L=1 Sanity

`fit-sh-l1` command는 여러 view cache를 같은 H5/scene 단위로 모아, triangle별 view direction sample에서 RGB coefficient를 fitting한다.

basis:

```text
[1, x, y, z]
```

cbox 5-view sanity dataset:

| 항목 | 값 |
| --- | --- |
| Scene | `output/stage0_cbox_5view/cbox_5view.json` |
| H5 | `output/stage0_cbox_5view/input.h5` |
| GT views | `output/stage0_cbox_5view/stage0_gt/view_0000.npz` ~ `view_0004.npz` |
| Cache | `output/stage0_latents_cbox_5view/` |
| SH output | `output/stage0_sh_l1_cbox_5view/stage0_cbox_5view_sh_l1.pt` |
| Fitted triangles | `340 / 5633` with `min_views=4` |

실행 결과:

```text
scene,view_caches,fitted_triangles,total_triangles
stage0_cbox_5view,5,340,5633
```

### 해석

결론:

```text
train-only mean baseline으로 고친 뒤에는
현재 4 train / 2 test split에서 latent-only, material-only, train-global mean이
거의 같은 total PSNR 범위에 있다.
view direction을 붙인 decoder는 train target fitting은 좋아졌지만
test image-space generalization은 오히려 낮아졌다.
```

세부 관찰:

- test total PSNR은 train-global mean `10.25`, latent-only `10.05`, material-only `10.01`, latent+view `9.50`이다.
- `D(h,n,v)`는 train target에서는 강하지만, single-view-per-scene split에서는 view/scene prior를 외우기 쉽다.
- SH L=1 fitting은 실제 multi-view cbox에서 동작했다. 다만 5개 view에서도 `min_views=4`를 만족하는 visible triangle은 340개라, SH 학습용 dataset은 view 수를 더 늘려야 한다.

다음 버전 후보:

```text
v0.6-sh-l1-decoder
```

목표:

```text
scene당 multi-view GT를 확장하고,
SH L=1 coefficient를 직접 예측하는 decoder를 학습/평가한다.
```

## v0.6 - cbox 30-view SH L=1 probe

### 목적

v0.6에서는 cbox 단일 scene을 30개 orbit view로 렌더링하고, train view 20개만 사용해 triangle별 SH L=1 coefficient target을 fitting했다. 이후 세 decoder probe를 비교했다.

```text
latent:          D(h_i) -> SH direct/indirect L=1
material-only:   D(m_i) -> SH direct/indirect L=1
latent+material: D(h_i, m_i) -> SH direct/indirect L=1
```

heldout 평가는 view `0020`~`0029`의 ID buffer gather로 predicted direct/indirect/total PNG/EXR를 재구성했다.

### 실행 산출물

| 경로 | 역할 |
| --- | --- |
| `output/stage0_cbox_30views/cbox_30views.json` | 30-view orbit camera config |
| `output/stage0_cbox_30views/stage0_gt/view_*.npz` | 30개 Stage 0 GT pass |
| `output/stage0_cbox_30views/cache_all/` | 30개 RenderFormer latent/material cache |
| `output/stage0_cbox_30views/cache_train/` | train view `0000`~`0019` cache |
| `output/stage0_cbox_30views/cache_test/` | heldout view `0020`~`0029` cache |
| `output/stage0_cbox_30views/latents_multiview.pt` | multi-view cache manifest |
| `output/stage0_cbox_30views/sh_l1_targets.pt` | train-only SH L=1 targets |
| `output/stage0_cbox_30views/sh_l1_summary.csv` | SH fitting summary |
| `output/stage0_cbox_30views/sh_l1_probe_metrics.csv` | coefficient-space probe metrics |
| `output/stage0_cbox_30views/heldout_renderings/` | heldout PNG/EXR/sheet outputs |
| `output/stage0_cbox_30views/heldout_view_metrics.csv` | heldout image-space metrics |

### SH Fitting

`min_views=6`을 유지한 결과:

| Scene | Train views | Fitted triangles | Total triangles |
| --- | ---: | ---: | ---: |
| `stage0_cbox_30views` | 20 | 396 | 5633 |

요청서의 희망 기준인 valid SH triangles `>=1000`에는 도달하지 못했다. cbox 30-view orbit이라도 현재 split triangle granularity에서는 충분히 많은 triangle이 6개 이상 view에서 관찰되지 않는다.

### Coefficient-Space Metrics

낮을수록 좋다.

| Method | direct SH MSE | indirect SH MSE | total SH MSE | direct dir MSE | indirect dir MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent | 0.00496 | 0.00225 | 0.00377 | 0.00497 | 0.00219 |
| material-only | 0.06944 | 0.00551 | 0.08395 | 0.06175 | 0.00499 |
| latent+material | 0.00441 | 0.00194 | 0.00361 | 0.00452 | 0.00193 |

coefficient-space에서는 latent와 latent+material이 material-only보다 명확히 좋다. 특히 direct directional coefficient에서 latent `0.00497`, latent+material `0.00452`, material-only `0.06175`로 차이가 크다.

### Heldout Image-Space Metrics

view `0020`~`0029` 평균 PSNR, 높을수록 좋다.

| Method | direct PSNR | indirect PSNR | total PSNR | direct leakage margin | indirect leakage margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| latent | 16.89 | 21.91 | 16.66 | 5.05 | 8.19 |
| material-only | 16.69 | 22.22 | 16.76 | 3.16 | 8.05 |
| latent+material | 16.90 | 22.04 | 16.69 | 5.05 | 8.32 |
| train SH mean | 14.21 | 19.73 | 13.90 | -0.06 | 6.33 |

heldout image-space total PSNR은 material-only `16.76`, latent+material `16.69`, latent `16.66` 순서다. 따라서 이번 설정에서는 "latent/latent+material이 material-only보다 heldout image-space 일반화가 좋다"고 말할 수 없다.

다만 train SH mean baseline보다는 모든 learned probe가 total PSNR에서 2.7 dB 이상 높다.

### 해석

- coefficient-space target fitting은 latent가 material-only보다 훨씬 강하다.
- image-space heldout gather에서는 material-only가 total PSNR에서 근소하게 앞선다.
- valid SH triangle이 396개로 낮아, image-space 평가는 많은 미관측/부정확 triangle과 blocky artifact 영향을 받는다.
- 다음 단계는 view 수를 더 늘리거나, `min_views=6`을 만족하는 triangle coverage를 늘리는 camera schedule을 다시 잡는 것이다.

### Debug Pass

추가 디버그에서 다음을 확인했다.

- `target_path`와 `feature_cache_path`의 H5는 모두 `output/stage0_cbox_30views/cbox_input.h5`로 일치한다.
- H5 triangle 배열과 split OBJ triangle 배열은 object별 offset/face count/max diff 기준으로 일치한다.
- Blender `Object Index -> triangle_id_buffer` mapping도 총 5633개로 H5 triangle 수와 일치한다.
- heldout `view_0020`은 visible pixel 중 fitted pixel이 66.2%뿐이고, visible unique triangle 1951개 중 fitted visible triangle은 379개다.
- heldout 전체 평균 fitted pixel ratio는 86.8%지만, unique fitted triangle coverage는 낮다.
- SH 음수 clamp mask도 존재하지만, 큰 검은 면의 주원인은 fitted coverage 부족이다.

feature normalization을 추가해 normalized probe를 재학습했다.

| Method | coefficient total SH MSE | heldout total PSNR |
| --- | ---: | ---: |
| latent norm | 0.00309 | 16.82 |
| material norm | 0.07238 | 16.92 |
| latent+material norm | 0.00326 | 16.82 |

산출물:

| 경로 | 역할 |
| --- | --- |
| `output/stage0_cbox_30views/probe_latent_to_sh_l1_norm.pt` | normalized latent probe |
| `output/stage0_cbox_30views/probe_material_to_sh_l1_norm.pt` | normalized material probe |
| `output/stage0_cbox_30views/probe_latent_material_to_sh_l1_norm.pt` | normalized latent+material probe |
| `output/stage0_cbox_30views/sh_l1_probe_metrics_norm.csv` | normalized coefficient metrics |
| `output/stage0_cbox_30views/heldout_renderings_norm_debug/` | fitted mask, negative mask, coefficient component images |
| `experiments/debug_stage0_alignment.py` | H5/split/mapping alignment diagnostic |

## v0.7 - interleaved split, fallback metrics, residual probe

### 변경점

- heldout 평가에서 unfitted triangle coefficient를 0으로 두지 않고 fallback으로 채우도록 수정했다.
- 기본 fallback은 material probe prediction이며, material extrapolation 폭주를 막기 위해 train SH target coefficient 범위로 clipping한다.
- heldout metric을 `all pixels`, `fitted pixels only`, `unfitted pixels only`로 분리했다.
- 모든 heldout view에 `fitted_mask.png`, `obs_count.png`, `negative_*_mask_*.png`, `(GT - pred) * 5` diff image를 저장한다.
- view split을 연속 split에서 interleaved split으로 변경했다.
- material baseline 위에 latent residual을 더하는 `train-sh-residual-probe`를 추가했다.

### Split

```text
test views  = 0,3,6,9,12,15,18,21,24,27
train views = 1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29
```

### 산출물

| 경로 | 역할 |
| --- | --- |
| `output/stage0_cbox_30views_interleaved/sh_l1_targets.pt` | interleaved train-only SH target |
| `output/stage0_cbox_30views_interleaved/probe_latent_to_sh_l1_norm.pt` | normalized latent SH probe |
| `output/stage0_cbox_30views_interleaved/probe_material_to_sh_l1_norm.pt` | normalized material SH probe |
| `output/stage0_cbox_30views_interleaved/probe_latent_material_to_sh_l1_norm.pt` | normalized latent+material SH probe |
| `output/stage0_cbox_30views_interleaved/probe_material_residual_latent_to_sh_l1_norm.pt` | material residual latent probe |
| `output/stage0_cbox_30views_interleaved/heldout_renderings_material_fallback_clipped_debug/` | material fallback/clipped evaluation images |
| `output/stage0_cbox_30views_interleaved/heldout_renderings_trainmean_fallback_debug/` | train mean fallback evaluation images |
| `output/stage0_cbox_30views_interleaved/heldout_view_metrics_material_fallback_clipped.csv` | material fallback metrics |
| `output/stage0_cbox_30views_interleaved/heldout_view_metrics_trainmean_fallback.csv` | train mean fallback metrics |

### 결과

SH fitting:

| Train views | Fitted triangles | Total triangles |
| ---: | ---: | ---: |
| 20 | 407 | 5633 |

Material fallback, clipped:

| Method | total all | total fitted only | total unfitted only |
| --- | ---: | ---: | ---: |
| latent | 15.02 | 21.38 | 3.74 |
| material-only | 15.08 | 20.70 | 3.74 |
| latent+material | 15.02 | 21.38 | 3.74 |
| material residual latent | 15.01 | 21.38 | 3.74 |
| train SH mean | 12.10 | 13.05 | 9.77 |

Train mean fallback:

| Method | total all | total fitted only | total unfitted only |
| --- | ---: | ---: | ---: |
| latent | 17.85 | 21.38 | 9.77 |
| material-only | 17.91 | 20.70 | 9.77 |
| latent+material | 17.86 | 21.38 | 9.77 |
| material residual latent | 17.84 | 21.38 | 9.77 |
| train SH mean | 12.10 | 13.05 | 9.77 |

### 해석

- fitted-only metric에서는 latent 계열이 material-only보다 좋다.
- all-pixel metric은 fallback 품질에 크게 좌우된다.
- material fallback은 0 fallback보다 낫지만, unfitted 영역에서는 train SH mean보다 낮다.
- material probe를 fitted triangle에서만 학습했기 때문에 모든 unfitted triangle으로 extrapolate하면 coefficient가 폭주할 수 있다. 그래서 clipping이 필요했다.
- 다음 단계는 SH coverage를 늘리거나, unfitted 영역 전용 fallback 모델을 별도로 학습하는 것이다.

## v0.8 - examples 15 scenes, 60 views

### 목적

SH L=1 coverage 병목을 확인하기 위해 `examples/`의 15개 scene을 60-view orbit으로 확장했다.

camera schedule:

```text
elevation = 10, 20, 30, 40 degrees
azimuth count per elevation = 15
total views = 60
```

split:

```text
test views = every 3rd view, indices where view_index % 3 == 0
train views = the other 40 views
```

### 산출물

| 경로 | 역할 |
| --- | --- |
| `output/stage0_examples_60views/manifest_60views.json` | 15 scene 60-view manifest |
| `output/stage0_examples_60views/<scene>/<scene>_60views.json` | scene별 60-view config |
| `output/stage0_examples_60views/<scene>/<scene>_input.h5` | scene별 RenderFormer input H5 |
| `output/stage0_examples_60views/<scene>/stage0_gt/view_*.npz` | scene별 60개 Stage 0 GT |
| `output/stage0_examples_60views/<scene>/sh_l1_targets.pt` | train 40-view SH L=1 target |
| `output/stage0_examples_60views/<scene>/feature_cache/*.pt` | scene당 1개 latent/material feature cache |
| `output/stage0_examples_60views/sh_l1_coverage_summary.csv` | SH coverage summary |

추가 스크립트:

| 파일 | 역할 |
| --- | --- |
| `experiments/prepare_stage0_60views.py` | examples scene을 60-view orbit config로 변환 |
| `experiments/fit_stage0_sh_l1_from_gt.py` | view별 full latent cache 없이 GT NPZ에서 직접 SH L=1 fitting |

### 실행 결과

15개 scene 모두:

```text
GT views: 60 / 60
feature cache: 1 / scene
SH target: generated
```

coverage:

| Scene | Fitted triangles | Total triangles | Ratio |
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

### 해석

- 60-view로 늘려도 일반 object scene의 `min_views=6` fitted triangle ratio는 대부분 3-13% 수준이다.
- `init-template`은 거의 배경/벽/조명만 있는 단순 scene이라 99% coverage가 나온다.
- `fox-in-the-wild`는 상대적으로 triangle 수가 적고 orbit visibility가 좋아 21%까지 올라간다.
- cbox 계열은 30-view cbox 396/5633에서 60-view cbox 438/5633으로 소폭 개선에 그쳤다.
- 단순히 orbit view 수를 늘리는 것보다, surface visibility를 보장하는 camera schedule 또는 per-surface sampling 전략이 필요하다.

주의:

- 처음에는 view별 full feature cache를 저장했지만 900 view에서 디스크가 가득 찼다.
- 이후 중복 cache를 제거하고, SH target은 GT NPZ에서 직접 fitting하며 feature는 scene당 1개만 저장하는 방식으로 바꿨다.
