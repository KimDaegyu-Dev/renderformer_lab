# RenderFormer Lab Repository Map and Stage 0 Plan

이 문서는 이 레포지토리의 레이어별 기능, 파일별 역할, 파일 구조, 그리고 direct / indirect radiance probing 실험인 Stage 0의 실행 단계와 현재 진행 상태를 정리한다.

## 1. 레포지토리의 큰 목적

RenderFormer는 triangle mesh scene을 입력으로 받아 global illumination이 반영된 이미지를 직접 예측하는 neural rendering pipeline이다.

핵심 구조는 2단계다.

1. View-independent stage
   - triangle sequence를 받아 triangle-to-triangle light transport를 모델링한다.
   - 출력은 register token과 per-triangle latent token이다.
   - Stage 0 probing 실험의 관찰 대상이다.

2. View-dependent stage
   - camera ray bundle token이 view-independent triangle token에 cross-attention한다.
   - 최종 image patch/RGB를 생성한다.

Stage 0의 질문은 다음이다.

```text
RenderFormer의 view-independent per-triangle latent token에서
direct / indirect radiance 정보를 작은 decoder로 분리해낼 수 있는가?
```

## 2. 최상위 파일 구조

```text
renderformer_lab/
  renderformer/                # 모델, 레이어, positional encoding, rendering pipeline
  scene_processor/             # JSON scene -> mesh/H5 변환, Blender reference rendering
  examples/                    # 테스트 scene JSON과 OBJ asset
  medias/                      # README용 이미지
  tmp/                         # 변환된 H5 예시 산출물
  output/                      # inference 결과 및 실험 산출물
  experiments/                 # Stage 0 probing 실험 스크립트
  docs/                        # 레포/실험 정리 문서
  infer.py                     # 단일 H5 RenderFormer inference
  batch_infer.py               # H5 폴더 batch inference
  render-images.sh             # 이미지 inference 예시 shell script
  render-videos.sh             # 비디오 inference 예시 shell script
  requirements.txt             # Python dependency 목록
```

## 3. 레이어별 기능

### 3.1 Scene/data layer

담당 디렉토리:

```text
scene_processor/
examples/
tmp/
```

역할:

- scene JSON을 읽는다.
- OBJ mesh를 transform/remesh/split한다.
- RenderFormer 입력 H5를 생성한다.
- 필요하면 Blender/Cycles로 reference image를 만든다.

중요 데이터 필드:

```text
triangles: [N, 3, 3]
vn:        [N, 3, 3]
texture:   [N, 13, 32, 32]
c2w:       [V, 4, 4]
fov:       [V]
```

`texture`의 13채널은 diffuse, specular, roughness, normal, emissive/irradiance 계열 material 정보를 triangle texture patch 형태로 담는다.

Stage 0에 추가로 필요한 GT 필드:

```text
triangle_id_buffer: [V, H, W] 또는 [H, W]
I_direct:           [V, H, W, 3] 또는 [H, W, 3]
I_indirect:         [V, H, W, 3] 또는 [H, W, 3]
I_total:            [V, H, W, 3] 또는 [H, W, 3]
```

현재 레포의 기본 H5에는 Stage 0 GT 필드가 없고, RenderFormer 입력 필드만 있다. 이번 cbox 실험에서는 RenderFormer 입력 H5와 Stage 0 GT를 분리해 저장했다.

```text
RenderFormer input: output/stage0_cbox/cbox_input.h5
Stage 0 GT:         output/stage0_cbox/stage0_gt/view_0000.npz
Y-flipped rerun:    output/stage0_cbox_yflip/stage0_gt/view_0000.npz
```

### 3.2 Model input encoding layer

담당 파일:

```text
renderformer/models/renderformer.py
renderformer/encodings/nerf_encoding.py
renderformer/encodings/rope.py
```

역할:

- triangle vertex position을 positional encoding한다.
- vertex normal을 optional encoding한다.
- 13채널 texture patch를 latent token으로 projection한다.
- register token과 triangle token을 하나의 sequence로 만든다.

핵심 함수:

```python
RenderFormer.construct_seq(...)
```

이 함수가 만드는 sequence는 다음 형태다.

```text
[register tokens, triangle tokens]
```

### 3.3 View-independent transport layer

담당 파일:

```text
renderformer/models/renderformer.py
renderformer/layers/attention.py
```

역할:

- triangle sequence에 self-attention transformer를 적용한다.
- triangle 간 light transport/global context를 latent에 섞는다.
- output의 triangle token 부분이 Stage 0의 probing 대상이다.

핵심 모듈:

```python
self.transformer = TransformerEncoder(...)
```

이번 실험을 위해 추가한 public hook:

```python
RenderFormer.extract_view_independent_latents(...)
```

반환값:

```text
latents:    [B, N, latent_dim]
valid_mask: [B, N]
```

기본적으로 register token은 제외하고 per-triangle token만 반환한다.

### 3.4 View-dependent rendering layer

담당 파일:

```text
renderformer/models/view_transformer.py
renderformer/layers/dpt.py
renderformer/utils/ray_generator.py
renderformer/utils/transform.py
```

역할:

- camera pose/fov에서 ray map을 만든다.
- ray patch token이 triangle latent sequence에 cross-attention한다.
- DPT head 또는 linear head로 RGB image를 만든다.

Stage 0에서는 이 레이어를 학습하지 않는다. 목적은 view-independent latent 자체를 probing하는 것이므로 RenderFormer 전체는 frozen 상태로 둔다.

### 3.5 Pipeline/inference layer

담당 파일:

```text
renderformer/pipelines/rendering_pipeline.py
infer.py
batch_infer.py
```

역할:

- Hugging Face pretrained model을 로드한다.
- H5 입력을 tensor로 변환한다.
- camera coordinate transform과 ray generation을 처리한다.
- RenderFormer 결과를 EXR/PNG로 저장한다.

## 4. 파일별 역할

### Root scripts

| 파일                     | 역할                                                             |
| ------------------------ | ---------------------------------------------------------------- |
| `infer.py`               | 단일 H5 scene을 RenderFormer로 렌더링하고 EXR/PNG를 저장한다.    |
| `batch_infer.py`         | 폴더 내 여러 H5를 batch로 렌더링하고 선택적으로 비디오를 만든다. |
| `render-images.sh`       | 예시 이미지 렌더링 명령 모음이다.                                |
| `render-videos.sh`       | 예시 비디오 렌더링 명령 모음이다.                                |
| `download_video_data.sh` | 비디오 예제 데이터를 다운로드한다.                               |
| `requirements.txt`       | 실행에 필요한 Python dependency 목록이다.                        |

### `renderformer/models`

| 파일                  | 역할                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `renderformer.py`     | RenderFormer 본체. triangle sequence 구성, view-independent transformer, view-dependent transformer 호출을 담당한다. |
| `view_transformer.py` | ray patch token과 triangle context token 간 cross-attention 및 image decoding을 담당한다.                            |
| `config.py`           | latent dim, layer 수, RoPE, texture channel, DPT decoder 등 모델 설정 dataclass다.                                   |

### `renderformer/layers`

| 파일           | 역할                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------ |
| `attention.py` | encoder/decoder transformer, self/cross attention, RoPE 적용, SDPA/FlashAttention fallback을 구현한다. |
| `dpt.py`       | view transformer의 multi-layer feature를 image로 복원하는 DPT head다.                                  |

### `renderformer/encodings`

| 파일               | 역할                                                             |
| ------------------ | ---------------------------------------------------------------- |
| `nerf_encoding.py` | NeRF-style sinusoidal positional encoding을 구현한다.            |
| `rope.py`          | triangle geometry에 대한 rotary positional embedding을 구현한다. |

### `renderformer/utils`

| 파일               | 역할                                                              |
| ------------------ | ----------------------------------------------------------------- |
| `ray_generator.py` | camera pose/fov/resolution에서 ray origin/direction map을 만든다. |
| `transform.py`     | world/camera coordinate 변환 유틸리티다.                          |

### `scene_processor`

| 파일               | 역할                                                                           |
| ------------------ | ------------------------------------------------------------------------------ |
| `scene_config.py`  | scene JSON schema dataclass다. object/material/camera 설정을 정의한다.         |
| `scene_mesh.py`    | OBJ 로드, normalize, transform, remesh, vertex color/material 처리를 수행한다. |
| `to_h5.py`         | split mesh와 scene config를 RenderFormer H5 입력으로 저장한다.                 |
| `convert_scene.py` | JSON scene을 mesh/H5로 변환하는 CLI entrypoint다.                              |
| `to_blend.py`      | Blender/Cycles scene을 구성하고 reference image 또는 blend 파일을 만든다.      |
| `to_blend_stage0.py` | Stage 0용 direct/indirect/combined/Object Index pass를 렌더링하고 triangle 평균 GT를 저장한다. |
| `remesh.py`        | mesh simplification/remeshing helper다.                                        |

### `experiments`

| 파일              | 역할                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------ |
| `stage0_probe.py` | Stage 0 latent 추출, triangle 평균 target 생성, direct/indirect MLP probe 학습을 담당한다. |

## 5. Stage 0 실험 설계

### 5.1 목표

Frozen RenderFormer의 view-independent per-triangle latent를 입력으로 작은 MLP decoder를 학습한다.

```text
h_i -> (L_direct_i, L_indirect_i)
```

출력 차원:

```text
direct RGB   3
indirect RGB 3
total        direct + indirect
```

### 5.2 최소 데이터셋

초기 권장 scene:

| 항목         | 값                      |
| ------------ | ----------------------- |
| Scene        | Cornell box류 단순 장면 |
| Material     | diffuse 위주            |
| Light        | area light 1개          |
| Triangle 수  | 256 ~ 1024              |
| Resolution   | 128 또는 256            |
| Component    | direct / indirect       |
| RenderFormer | frozen                  |
| 학습 대상    | MLP decoder만           |

### 5.3 GT 생성

가장 빠른 구현은 triangle ID buffer를 이용해 visible triangle 평균 target을 만드는 방식이다.

```text
L_direct_tri[i]   = mean(I_direct[p]   for p where triangle_id[p] == i)
L_indirect_tri[i] = mean(I_indirect[p] for p where triangle_id[p] == i)
```

visible pixel 수가 너무 적은 triangle은 loss에서 제외한다.

기본 threshold:

```text
min_pixels = 8
```

### 5.4 Decoder

`experiments/stage0_probe.py`의 기본 MLP:

```text
Linear(latent_dim, 512)
GELU
Linear(512, 256)
GELU
Linear(256, 6)
Softplus
```

`latent_dim`은 모델 설정에 따라 다르다. 이번 `microsoft/renderformer-v1.1-swin-large` 실험에서는 extracted latent shape가 `[5633, 1024]`였다.

### 5.5 Loss

```text
L_comp = L1(pred_direct, direct) + L1(pred_indirect, indirect)
L_sum  = L1(pred_direct + pred_indirect, total)
L      = L_comp + lambda_sum * L_sum
```

기본값:

```text
lambda_sum = 0.5
```

### 5.6 Metrics

현재 스크립트가 출력하는 지표:

```text
direct_psnr
indirect_psnr
total_psnr
```

이미지 재구성 평가에서 추가한 지표:

```text
direct_psnr_img
indirect_psnr_img
total_psnr_vs_direct_plus_indirect_img
direct_cross_psnr_img
indirect_cross_psnr_img
direct_leakage_margin_db
indirect_leakage_margin_db
```

component leakage margin은 자기 component와 반대 component의 PSNR 차이로 계산한다.

```text
direct_leakage_margin_db   = PSNR(pred_direct, GT_direct) - PSNR(pred_direct, GT_indirect)
indirect_leakage_margin_db = PSNR(pred_indirect, GT_indirect) - PSNR(pred_indirect, GT_direct)
```

## 6. 실행 방법

### 6.1 환경 준비

Windows에서 `renderformer-liger-kernel` 또는 FlashAttention이 설치되지 않아도 attention 구현은 SDPA로 fallback된다.

### 6.2 RenderFormer 입력 H5 생성

```bash
conda run -n renderformer python scene_processor/convert_scene.py examples/cbox.json \
  --mesh_path output/stage0_cbox/cbox.obj \
  --output_h5_path output/stage0_cbox/cbox_input.h5
```

### 6.3 Stage 0 GT 생성

Stage 0 supervised probe에는 다음 필드 또는 그로부터 계산된 triangle 평균 target이 필요하다.

```text
triangle_id_buffer
I_direct
I_indirect
I_total
```

이번 실험에서는 `scene_processor/to_blend_stage0.py`로 Blender/Cycles pass를 렌더링했다. Blender 내 `h5py`가 없는 환경에서는 H5 대신 NPZ를 저장하도록 fallback을 사용했다.

```bash
blender --background --factory-startup --python-expr "..."  # runpy로 to_blend_stage0.py 실행
```

실제 인자:

```text
scene_config: examples/cbox.json
output_dir:   output/stage0_cbox
mesh_path:    output/stage0_cbox/cbox.obj
resolution:   128
spp:          64
flags:        --stage0_gt --stage0_split_triangles
```

생성된 GT:

```text
output/stage0_cbox/stage0_gt/view_0000.npz
```

카메라 local Y/up 방향 보정을 적용한 재렌더링은 다음 flag를 추가해 생성했다.

```text
--stage0_flip_camera_y
```

Y-flipped GT:

```text
output/stage0_cbox_yflip/stage0_gt/view_0000.npz
```

### 6.4 Frozen view-independent latent 추출

```bash
conda run -n renderformer python -B experiments/stage0_probe.py extract-latents \
  --h5_glob "output/stage0_cbox/cbox_input.h5" \
  --gt_h5_glob "output/stage0_cbox/stage0_gt/view_0000.npz" \
  --output_dir output/stage0_latents_cbox \
  --model_id microsoft/renderformer-v1.1-swin-large \
  --device cuda \
  --precision float16
```

CPU에서 smoke test를 할 경우:

```bash
conda run -n renderformer python -B experiments/stage0_probe.py extract-latents \
  --h5_glob "tmp/cbox/*.h5" \
  --output_dir output/stage0_latents \
  --device cpu \
  --precision float32
```

### 6.5 Probe 학습

GT가 포함된 latent cache가 생성된 뒤 실행한다.

```bash
conda run -n renderformer python -B experiments/stage0_probe.py train-probe \
  --cache_glob "output/stage0_latents_cbox/*.pt" \
  --epochs 300 \
  --batch_size 512 \
  --lr 1e-4 \
  --output_model output/stage0_probe_cbox/probe_lr1e-4.pt \
  --device cuda
```

## 7. 현재 진행 상태

완료:

- 레포 구조와 모델 레이어 분석
- view-independent latent extraction hook 추가
- Stage 0 MLP probe 스크립트 추가
- 레포 구조/파일 역할/실험 계획 문서화
- cbox 단일 scene에 대해 Stage 0 GT 생성
- cbox RenderFormer input H5 생성
- frozen VI latent cache 생성
- direct/indirect MLP probe 학습 실행
- ID buffer gather 기반 predicted direct/indirect/total image 재구성
- GT/predicted pass PNG/EXR 및 comparison sheet 저장
- Stage 0 GT 카메라 local Y/up 축 반전 재렌더링
- material-only baseline 추가 및 학습/평가
- triangle mean baseline 추가 및 평가
- latent/material/triangle mean baseline 비교 CSV 저장
- component leakage metric 추가
- cbox 외 단순 scene 5개를 포함한 6-scene Stage 0 dataset 생성
- 4 train / 2 test split에서 material-only baseline 대비 일반화 성능 비교

초기 cbox 실험 요약:

```text
scene: examples/cbox.json
resolution: 128
spp: 64
triangles: 5633
visible/supervised triangles: 394
valid visible pixels: 13326
latent shape: [5633, 1024]
best probe checkpoint: output/stage0_probe_cbox/probe_lr1e-4.pt
final metrics: direct_psnr=30.07, indirect_psnr=15.71, total_psnr=31.83
gather eval output: output/stage0_eval_cbox/
```

현재 y-flipped cbox 실험 요약:

```text
scene: examples/cbox.json
resolution: 128
spp: 64
triangles: 5633
visible/supervised triangles: 397
valid visible pixels: 13376
latent shape: [5633, 1024]
latent checkpoint: output/stage0_probe_cbox_yflip/probe_latent_lr1e-4.pt
material-only checkpoint: output/stage0_probe_cbox_yflip/probe_material_lr1e-4.pt
latent eval output: output/stage0_eval_cbox_yflip_latent/
material eval output: output/stage0_eval_cbox_yflip_material/
baseline comparison: output/stage0_compare_cbox_yflip/baseline_comparison_metrics.csv
```

baseline 비교 결과:

| Method | direct PSNR | indirect PSNR | total PSNR vs direct+indirect |
| --- | ---: | ---: | ---: |
| RenderFormer latent probe | 15.75 | 21.59 | 16.77 |
| Material-only baseline | 22.22 | 21.41 | 22.78 |
| Triangle mean baseline | 10.74 | 18.86 | 12.12 |

해석:

- RenderFormer latent probe는 triangle mean baseline보다 direct/indirect/total 모두 높다.
- 하지만 단일 cbox, 단일 view, train=test overfit 조건에서는 material-only baseline이 direct와 total에서 더 높다.
- 현재 결과만으로는 RenderFormer latent가 material-only보다 낫다고 말할 수 없다. 다음 비교는 여러 scene/view split에서 진행해야 한다.

multi-scene v1 실험 요약:

```text
dataset root: output/stage0_multiscene_v1/
cache root: output/stage0_latents_multiscene_v1/
train scenes: cbox, cbox-bunny, cbox-teapot, constant-width
test scenes: compose-scene, shader-ball
resolution: 128
spp: 64
camera fix: --stage0_flip_camera_y
latent checkpoint: output/stage0_probe_multiscene_v1/probe_latent_lr1e-4.pt
material checkpoint: output/stage0_probe_multiscene_v1/probe_material_lr1e-4.pt
summary CSV: output/stage0_compare_multiscene_v1/split_summary_metrics.csv
per-scene CSV: output/stage0_compare_multiscene_v1/per_scene_metrics.csv
test image outputs: output/stage0_eval_multiscene_v1/
```

split 평균 결과:

| Split | Method | direct PSNR | indirect PSNR | total PSNR | direct leakage margin | indirect leakage margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | RenderFormer latent | 11.73 | 21.96 | 14.35 | 7.70 | 13.43 |
| train | Material-only | 12.77 | 24.87 | 15.49 | 7.88 | 16.08 |
| train | Triangle mean | 12.03 | 26.48 | 15.15 | 1.69 | 17.73 |
| test | RenderFormer latent | 9.79 | 17.06 | 10.05 | 5.99 | 6.23 |
| test | Material-only | 9.30 | 21.00 | 10.01 | 7.25 | 10.08 |
| test | Triangle mean | 12.85 | 31.35 | 13.92 | -1.77 | 20.96 |

multi-scene v1 해석:

- test split에서 latent probe는 material-only 대비 direct와 total이 아주 근소하게 높다.
- material-only baseline은 indirect와 leakage margin에서 더 안정적이다.
- total PSNR 차이가 0.04 dB 수준이라, 현재 설정에서는 latent가 material-only보다 일반화 성능이 명확히 높다고 말하기 어렵다.
- triangle mean baseline은 현재 각 평가 scene의 GT 평균을 쓰는 per-scene mean baseline이라 일반화 baseline이라기보다 낮은 주파수/scene 평균 sanity check에 가깝다.

multi-scene v2 실험 요약:

```text
cache root: output/stage0_latents_multiscene_v2/
train cache: output/stage0_latents_multiscene_v2_train/
test cache: output/stage0_latents_multiscene_v2_test/
latent-view checkpoint: output/stage0_probe_multiscene_v2/probe_latent_view_lr1e-4.pt
summary CSV: output/stage0_compare_multiscene_v2/split_summary_metrics.csv
per-scene CSV: output/stage0_compare_multiscene_v2/per_scene_metrics.csv
SH L=1 cbox sanity: output/stage0_sh_l1_cbox_5view/sh_l1_summary.csv
```

변경점:

- triangle mean baseline을 test scene 평균이 아닌 train-only global mean baseline으로 교체했다.
- cache에 `triangle_normals`, `view_dirs`, `latent_view_features = [latent, normal, view_dir]`를 추가했다.
- `D(h_i, n_i, v_i) -> L_i` probe를 학습했다.
- multi-view cache에서 per-triangle SH L=1 coefficient를 fitting하는 `fit-sh-l1` command를 추가했다.

split 평균 결과:

| Split | Method | direct PSNR | indirect PSNR | total PSNR | direct leakage margin | indirect leakage margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | RenderFormer latent | 11.73 | 21.96 | 14.35 | 7.70 | 13.43 |
| train | RenderFormer latent + view | 12.66 | 19.37 | 14.43 | 8.20 | 10.38 |
| train | Material-only | 12.77 | 24.87 | 15.49 | 7.88 | 16.08 |
| train | Train-global mean | 10.73 | 23.42 | 13.44 | 3.10 | 14.58 |
| test | RenderFormer latent | 9.79 | 17.06 | 10.05 | 5.99 | 6.23 |
| test | RenderFormer latent + view | 9.68 | 13.66 | 9.50 | 6.03 | 2.50 |
| test | Material-only | 9.30 | 21.00 | 10.01 | 7.25 | 10.08 |
| test | Train-global mean | 9.98 | 20.05 | 10.25 | 3.58 | 9.01 |

multi-scene v2 해석:

- train-only global mean baseline을 쓰면 test scene 평균 누수가 제거된다.
- 현재 test total PSNR은 train-global mean `10.25`, latent-only `10.05`, material-only `10.01`, latent+view `9.50` 순서다.
- `D(h_i, n_i, v_i)`는 train triangle target fitting은 좋아졌지만, 현재 4/2 scene split의 image-space test 일반화는 latent-only보다 낮다.
- view direction을 넣은 decoder는 단일 view/적은 scene에서 쉽게 scene-specific 방향 prior를 외우는 것으로 보인다.
- cbox 5-view sanity에서는 SH L=1 fitting command가 5개 view cache에서 340/5633 triangle의 coefficient를 fitting했다.

multi-view SH L=1 v0.6 실험 요약:

```text
GT root: output/stage0_cbox_30views/stage0_gt/
cache root: output/stage0_cbox_30views/cache_all/
train cache: output/stage0_cbox_30views/cache_train/
heldout cache: output/stage0_cbox_30views/cache_test/
SH target: output/stage0_cbox_30views/sh_l1_targets.pt
coefficient metrics: output/stage0_cbox_30views/sh_l1_probe_metrics.csv
heldout renderings: output/stage0_cbox_30views/heldout_renderings/
heldout metrics: output/stage0_cbox_30views/heldout_view_metrics.csv
```

결과:

| 평가 | latent | material-only | latent+material | train SH mean |
| --- | ---: | ---: | ---: | ---: |
| coefficient total SH MSE | 0.00377 | 0.08395 | 0.00361 | - |
| coefficient direct dir MSE | 0.00497 | 0.06175 | 0.00452 | - |
| heldout total PSNR | 16.66 | 16.76 | 16.69 | 13.90 |

해석:

- 30-view cbox에서 train view 20개만 사용해 SH L=1 target을 fitting했다.
- `min_views=6` 기준 fitted triangle은 396/5633으로, 목표였던 1000개 이상에는 못 미쳤다.
- coefficient-space에서는 latent와 latent+material이 material-only보다 훨씬 좋다.
- heldout image-space total PSNR은 material-only가 근소하게 앞선다.
- 따라서 현재 단계의 결론은 "latent가 SH coefficient를 잘 예측하지만, cbox heldout image reconstruction 일반화 우위는 아직 확인되지 않았다"이다.

주의:

- 이번 결과는 단일 scene, 단일 view, 64 spp GT에 대한 sanity check다.
- train/test split이 없으므로 일반화 성능이 아니라 pipeline 연결성과 overfit 가능성을 확인한 결과다.
- 기본 `lr=1e-3`는 초반 이후 정체했고, `lr=1e-4`가 훨씬 안정적으로 학습됐다.

## 8. 다음 구현 순서

1. multi-scene dataset을 scene당 5개 이상 view로 확장한다.
2. SH L=1 target을 학습 대상으로 쓰는 decoder head를 추가한다.
3. train-global mean보다 나은 일반화 성능을 만들 수 있도록 scene/view split을 키운다.
4. component leakage를 줄이는 loss 또는 probe head 구조를 실험한다.
5. direct-only cel shading sanity check를 추가한다.
