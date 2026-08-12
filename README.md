# DiffuBind：面向跨设备屏摄的载体锚定扩散水印

**DiffuBind: Carrier-Anchored Diffusion Watermarking for Cross-Device Screen Recapture**

DiffuBind 是一个面向显示器、OLED、LED 大屏和投影设备屏摄场景的扩散水印实验框架。系统以已有载体图像（carrier / cover）为锚点，通过 image-to-image diffusion 进行内容保持的重构，并在反向去噪过程中同时注入全局与空间二进制水印条件。当前正式配置使用 `128×128` RGB 图像和 `30-bit` 消息。

> 当前仓库不包含训练数据、checkpoint 或最终实验日志。本文只描述代码与正式 YAML 中已经实现的功能，不报告未经复现的 ACC、BER、PSNR、SSIM、LPIPS 或真实屏摄结果。Experimental results will be added after the final evaluation.

## 项目概述

传统的扩散生成从近似纯高斯噪声开始；DiffuBind 的训练与推理都以载体图像为起点：

```text
cover image
    -> forward diffusion to t_start
    -> carrier-conditioned reverse diffusion
    -> watermarked image
```

因此，本项目的核心不是“从噪声生成一张带水印的新图”，而是 **carrier-anchored / carrier-conditioned reconstruction**：原图既决定反向过程的内容条件，也决定水印空间条件和允许的残差预算。

当前主链路由四部分组成：

1. `WatermarkConditionedUNet`：以 `x_t`、cover 和水印为条件预测扩散噪声；
2. Content-Aware Binding：从 cover 的边缘与纹理构造 soft mask；
3. Carrier-Aware Residual Constraint：按内容 mask 限制 `pred_x0 - cover`；
4. `ResidualMultiScaleWatermarkDecoder`：从 clean 或屏摄退化图像恢复消息。

## 研究动机

跨设备屏摄会同时引入几何、显示、光学、采样和传感器失真。若水印只依赖不受约束的加性扰动，容易在平坦区域形成彩色偏置、条纹或窄带频谱峰；若只追求 clean 解码，又难以覆盖不同设备的退化分布。DiffuBind 因此将以下目标联合起来：

- 以 cover 约束图像内容与整体外观；
- 将更多水印承载能力分配给边缘和局部纹理；
- 同时提供 global message conditioning 与 spatial message conditioning；
- 用可微屏摄近似训练 Decoder，并小幅适配 Encoder；
- 同时优化平均攻击性能和最差设备性能；
- 使用固定“设备 × 强度”矩阵稳定选择 checkpoint。

## 方法总览

```text
Carrier Image
      |
      +----------------------> Multi-scale Edge / Texture Analysis
      |                                      |
      |                                 Content Mask
      |                                      |
      v                                      v
Forward Diffusion                     Spatial WM Gating
      |
      v
     x_t
      |
      +---- Carrier Image
      +---- Global WM Embedding ----> fused with timestep embedding
      +---- Spatial WM Map ---------> gated by Content Mask
      |                                      |
      +------------------ concat ------------+
                             |
                             v
                    Conditional U-Net
                             |
                             v
                         pred_x0
                             |
                             v
              Carrier-Anchored Residual Constraint
                             |
                             v
                    Watermarked Image
                      |             |
                      |       Screen Recapture
                      |      PIMoG / OLED / LED /
                      |          Projector
                      v             v
                   Decoder       Decoder
                      |             |
                      +------v------+
                             |
                     Recovered Message
```

训练阶段使用低时间步的单步 `pred_x0` 作为可微代理；周期采样、独立采样和合成鲁棒性评估才执行从 `t_start` 到 0 的完整 DDPM 反向轨迹。两类结果不可直接混为一谈。

## 载体锚定扩散

正式配置使用 1000-step linear beta schedule。训练包含两条 U-Net 分支：

- **Diffusion branch**：在完整时间步 `[0, 1000)` 采样 `t_diff`，以 MSE 学习噪声预测；
- **Watermark branch**：在 `[wm_t_min, wm_t_max) = [0, 200)` 采样 `t_wm`，从预测噪声解析 `pred_x0`，再计算水印与图像约束。

推理时，给 cover 在 `t_start - 1` 处加噪，再逐步执行 `t_start` 次 DDPM reverse update。`t_start` 同时决定前向扰动幅度和实际反向步数；当前正式训练设置为 `train_t_start: 200`。

## 内容感知水印绑定

### 基于载体内容的掩码

`models/watermark_residual.py` 从 `[0,1]` cover 的亮度通道构造内容相关 soft allowance：

1. 在多个尺度计算 Sobel edge response；
2. 在多个局部窗口计算 local standard deviation；
3. 分别归一化后按 `edge_weight` / `texture_weight` 融合；
4. 对每张图使用分位数阈值，使 support area 在不同 cover 之间可比较；
5. 用 sigmoid 和 `mask_temperature` 得到连续 soft mask。

高 allowance 表示边缘或纹理较丰富、相对适合承载水印；平坦区域 allowance 较低。mask 从 cover 计算并 detach，不反向修改 cover 分析过程。

### 全局—空间双路水印条件

30-bit 消息通过两条条件路径进入 U-Net：

```text
Global path:
watermark bits -> watermark_mlp -> 256-D embedding
               -> add to timestep embedding

Spatial path:
watermark bits -> watermark_map_mlp -> 4 x 16 x 16 map
               -> bilinear interpolation to 4 x 128 x 128
               -> content-aware gating
```

空间门控为：

```text
gate = wm_map_flat_floor + (1 - wm_map_flat_floor) * content_mask
gated_wm_map = wm_map * gate
```

正式配置同时启用两条路径。U-Net 的空间输入为：

```text
x_t       : 3 channels
cover     : 3 channels
wm_map    : 4 channels
----------------------
total     : 10 channels
```

这不是单一 watermark embedding，而是 **Global + Spatial Dual Watermark Conditioning**。

## 载体感知残差约束

模型从噪声预测恢复候选 `pred_x0` 后，不直接把任意扰动写入 cover。代码先在 `[0,1]` 域计算 raw residual，再按 content allowance 分配空间预算：

```text
budget = flat_budget
       + (texture_budget - flat_budget) * allowance^mask_power

bounded_delta = budget * tanh(raw_delta / max(budget, eps))
watermarked   = clamp(cover + bounded_delta, 0, 1)
```

等价地，平坦区域只允许很小的残差；边缘和纹理区域可以使用相对更大的预算。平滑 `tanh` 投影在训练和完整 DDPM 嵌入中均可参与约束，避免把水印理解为 unrestricted additive perturbation。

## 水印解码器

正式配置使用 `ResidualMultiScaleWatermarkDecoder`：

```text
RGB image [-1,1]
    -> stem: Conv / GroupNorm / SiLU
    -> residual blocks + stride-2 downsampling
    -> 32x32, 16x16, 8x8 feature maps
    -> global average pooling at three scales
    -> feature concatenation
    -> Linear / SiLU / Dropout / Linear
    -> 30 raw logits
```

Decoder 内部不使用 sigmoid，也不做图像归一化：

- 训练：`BCEWithLogitsLoss(logits, bits)`；
- 推理：`sigmoid(logits) > 0.5`。

仓库保留 `decoder.type: simple` 作为兼容/消融入口，但两份正式 YAML 都使用 `residual_multiscale`。

## 跨设备屏摄退化模拟

所有正式 Noise Layer 均使用统一的 `[0,1] -> [0,1]` 接口，并在 FP32 中执行。

### PIMoG / LCD 类屏幕

`PIMoGLayer` 保留 PIMoG 风格的 perspective、illumination、moiré 与 Gaussian noise 链路。内部历史实现使用 `[-1,1]`；adapter 负责从统一 `[0,1]` 输入转换到旧范围，再把输出转换回 `[0,1]`。

### OLED 屏幕

当前 OLED 链路包含：

- OLED tone response、gamma、contrast、saturation 与 highlight/black response；
- PenTile 或 stripe subpixel pattern；
- subpixel emission spread、display blur 与 camera blur；
- perspective；
- PWM / rolling-shutter banding；
- viewing-angle color shift；
- signal-dependent sensor noise 与 Gaussian noise；
- motion blur；
- reflection / haze；
- optional resampling 与 differentiable JPEG proxy。

正式 Stage 2 配置关闭 `enable_resample` 和 `use_jpeg`，其余列出的主模块按各自开关与概率运行。

### LED 大屏

当前 LED 链路包含：

- low effective resolution / downsampling；
- LED bead / pixel-grid structure；
- RGB triplet 或 mono-dot emission；
- bloom 与亮度/颜色变化；
- scanline；
- moiré；
- perspective；
- camera blur、resampling 与 sensor noise。

正式配置使用 `severity: medium`、`rgb_triplet` 与 Gaussian dot，并启用 scanline、moiré 和 perspective。

### 投影设备

当前 Projector 链路是受 projector-camera forward modeling / DeProCams 思路启发的轻量可微近似，不能视为 DeProCams 的等价实现。它可模拟：

- projector gamma；
- radial brightness falloff 与 hotspot；
- projection surface texture；
- spatially varying defocus；
- keystone / perspective；
- ambient light 与 contrast reduction；
- color gain / bias；
- camera sensor noise；
- optional pixel grid、moiré 与 lens distortion。

正式 Stage 2 配置关闭 pixel grid、moiré 和 lens distortion。

### 张量数值范围

这是不可省略的工程边界：

```text
Diffusion model / Decoder : [-1,1]
Noise Layer               : [0,1]

pred_x0 [-1,1]
    -> (x + 1) / 2
    -> degradation [0,1]
    -> 2x - 1
    -> Decoder [-1,1]
```

## 训练目标

训练目标不是简单罗列多个 loss，而是分别约束通信、保真、定位与结构伪影：

| 目标 | 作用 |
|---|---|
| diffusion noise MSE | 约束完整时间步的扩散噪声预测，避免模型只适配低时间步水印分支 |
| image L1 | 约束 `pred_x0` 与 cover 的整体保真度 |
| clean watermark BCE | 建立并保持未退化图像上的消息写入/读取通道 |
| degraded watermark BCE | 使消息经过屏摄近似后仍可恢复 |
| delta / TV / top-k residual terms | 分别约束残差幅度、局部高频变化和稀疏尖峰；正式配置中部分权重为 0，仅保留实现与日志 |
| channel balance | 惩罚 RGB 通道残差能量标准差，抑制 green/purple 等单通道偏置 |
| region energy-ratio loss | 要求残差能量集中于 cover-derived edge/texture support，并限制 support 外能量 |
| spectral peak loss | 抑制少数窄带 FFT 峰值 |
| directional anisotropy loss | 抑制过强水平/垂直方向性条纹 |
| multi-attack mean/worst BCE | 同时覆盖平均鲁棒性与最差设备鲁棒性 |

`cross_image_correlation`、FFT mid-band ratio、inside/outside energy ratio 等还会作为 validation diagnostics 记录；其中 cross-image correlation 当前是诊断指标，不直接进入 loss。

## 训练策略

### 第一阶段：干净图像上的载体绑定

配置：`configs/watermark_stage1.yaml`

Stage 1 使用 `noise_layer.type: none`，完整训练 Encoder 与 Decoder。它由三个手动阶段组成，必须依次执行 `warmup -> balance -> full`。当前提交的 YAML 是 `train.stage: full`，因此首次从零训练前必须先改为 `warmup`；`balance` 和 `full` 必须从前一阶段 checkpoint 通过 `--init_from` 初始化。

| 阶段 | 作用 | `wm_map_flat_floor` | flat / texture budget | `mask_power` | mask support / temperature | inside / outside target | spectral |
|---|---|---:|---:|---:|---:|---:|---|
| Warmup | 建立 clean watermark communication，让 Encoder 学会写入、Decoder 学会读取 | 0.20 | 0.012 / 0.060 | 1.0 | 0.30 / 0.05 | 0.80 / 0.20 | off |
| Balance | 提高画质、收紧空间预算、加强 edge/texture localization | 0.10 | 0.007 / 0.050 | 1.6 | 0.27 / 0.04 | 0.82 / 0.18 | step 2000 起 warmup |
| Full | 联合优化 clean ACC、carrier anchoring、RGB balance、方向与 FFT 伪影 | 0.03 | 0.003 / 0.040 | 2.8 | 0.24 / 0.03 | 0.85 / 0.15 | 从 step 0 开启 |

不同阶段的 edge/texture 尺度与融合权重：

| 阶段 | edge scales | texture scales | edge / texture weight |
|---|---|---|---:|
| Warmup | `[1,3,5]` | `[3,5,7]` | 0.55 / 0.45 |
| Balance | `[1,3]` | `[3,5]` | 0.65 / 0.35 |
| Full | `[1,3]` | `[3,5]` | 0.70 / 0.30 |

Warmup 使用固定主权重：

```text
lambda_diff=.01, lambda_img=.1, lambda_wm=20, lambda_region=.10
```

Balance 的 loss schedule 为：

| global step | `lambda_diff` | `lambda_img` | `lambda_wm` | `lambda_region` |
|---:|---:|---:|---:|---:|
| 0–2999 | 0.01 | 0.1 | 20 | 0.20 |
| 3000–5499 | 0.05 | 0.5 | 12 | 0.35 |
| 5500+ | 0.10 | 1.0 | 8 | 0.50 |

Full 使用：

```text
lambda_diff=.3, lambda_img=3, lambda_wm=3, lambda_delta=.05,
lambda_channel=.2, lambda_region=.5
```

### 第二阶段：跨设备鲁棒性训练

配置：`configs/watermark_stage2.yaml`

Stage 2 的目的不是重新学习一套水印嵌入，而是在尽量保留 Stage 1 carrier-anchored structure 的基础上增强跨设备屏摄鲁棒性。新实验必须用 `--init_from` 从 Stage 1 Full checkpoint 初始化；它只加载 Encoder/Decoder 权重，并新建 Stage 2 AdamW。`--resume` 仅用于继续同一个已中断的 Stage 2 实验。

当前 partial Encoder fine-tuning 精确解冻：

- `watermark_mlp`；
- `watermark_map_mlp`（`freeze_watermark_map_mlp: false`，当前允许更新）；
- U-Net 最后 3 个 `output_blocks`；
- U-Net 最终 `out`；
- 完整 Decoder。

Encoder 其余部分冻结并保持 eval mode。基础 Encoder LR 为 `1e-6`，Decoder LR 为 `2e-5`，二者都会乘以当前 curriculum 的 `lr_scale`。

Stage 2 使用比 Stage 1 Full 略宽松的 carrier budget，以容纳鲁棒性适配：

```text
wm_map_flat_floor       = 0.07
flat / texture budget  = 0.0035 / 0.050
mask_power              = 2.1
edge scales             = [1,3,5,7]
texture scales          = [3,5,7,9]
edge / texture weight   = 0.60 / 0.40
target_mask_area        = 0.31
mask_temperature        = 0.048
inside / outside target = 0.80 / 0.20
```

稳定性设置包括 AMP、`amp_init_scale=1024`、`amp_growth_interval=2000`、`amp_min_scale=1`、恢复时低于 16 则重置 scaler、`max_grad_norm=1.0` 和最多 5 次连续 non-finite 保护。代码会在 forward、gradient 或 optimizer step 出现非有限值时跳过污染更新；连续达到阈值则停止训练，避免保存污染 checkpoint。

Stage 2 还使用三段视觉 loss schedule：

| global step | `lambda_diff` | `lambda_img` | `lambda_channel` | `lambda_region` |
|---:|---:|---:|---:|---:|
| 0–5999 | 0.10 | 3.00 | 0.20 | 0.35 |
| 6000–17999 | 0.10 | 2.75 | 0.20 | 0.30 |
| 18000+ | 0.10 | 2.50 | 0.15 | 0.275 |

### 七段式退化课程

下表由当前 `train.noise_curriculum` 逐项整理。概率顺序与 candidates 一致。

| Phase / step | candidates | degradation probabilities | `apply_prob` | strength | `lambda_wm_clean` | `lambda_wm_degraded` | detach degraded from model | `lr_scale` |
|---|---|---|---:|---:|---:|---:|---|---:|
| 1 / 0–1999 | Projector, OLED | 0.50, 0.50 | 0.30 | 0.25 | 1.00 | 0.50 | true | 0.50 |
| 2 / 2000–5999 | Projector, OLED, PIMoG | 0.35, 0.35, 0.30 | 0.50 | 0.40 | 0.90 | 1.00 | false | 0.75 |
| 3 / 6000–11999 | PIMoG, OLED, LED, Projector | 0.30, 0.20, 0.35, 0.15 | 0.70 | 0.60 | 0.75 | 1.50 | false | 1.00 |
| 4 / 12000–17999 | PIMoG, OLED, LED, Projector | 0.33, 0.15, 0.37, 0.15 | 0.85 | 0.75 | 0.60 | 1.50 | false | 1.00 |
| 5 / 18000–23999 | PIMoG, OLED, LED, Projector | 0.35, 0.10, 0.40, 0.15 | 1.00 | 0.90 | 0.50 | 1.75 | false | 1.00 |
| 6 / 24000–27999 | PIMoG, OLED, LED, Projector | 0.35, 0.10, 0.40, 0.15 | 1.00 | 1.00 | 0.40 | 2.00 | false | 1.00 |
| 7 / 28000+ | PIMoG, OLED, LED, Projector | 0.35, 0.10, 0.40, 0.15 | 1.00 | 0.95 | 0.40 | 1.75 | false | 0.50 |

Phase 1 的 detach 只切断 degraded 分支经退化回到 Encoder 的梯度；clean watermark 分支仍可训练当前解冻的 Encoder 参数。

## 多攻击鲁棒优化

Stage 2 启用：

```yaml
multi_attack:
  enabled: true
  attacks_per_batch: 4
  lambda_mean: 0.30
  lambda_worst: 0.70
```

同一个 `pred_x0` 会在一次 batch 中分别接受多个不同退化。候选攻击不放回采样，实际数量为 `min(attacks_per_batch, 当前候选数)`，所以 Phase 1/2/3–7 分别最多执行 2/3/4 个攻击。degraded watermark loss 为：

```text
L_degraded = 0.30 * mean(L_attack_i)
           + 0.70 * max(L_attack_i)
```

这兼顾 average robustness 与 worst-device robustness。

不要混淆两个概念：

- `MixedNoiseLayer.forward()`：一次调用只为整个 batch 选择一种 degradation；
- `multi_attack`：对同一 watermarked image 多次调用指定的不同 degradation，再聚合 loss。

## 验证与检查点选择

训练期 validation 使用固定消息与 RNG，在低时间步生成 **single-step `pred_x0` surrogate**，而不是运行完整 200-step DDPM embedding。它同时记录：

- clean bit ACC 与 BCE；
- curriculum 强度下的 per-device ACC、BCE、SSIM；
- macro degraded ACC、worst-device ACC；
- watermarked PSNR、SSIM、MAE；
- top-k residual、TV、RGB channel imbalance；
- mask area、inside/outside energy ratio；
- directional ratio、FFT peak/mid-band ratio、cross-image correlation。

Stage 2 每个 validation epoch 还运行固定矩阵：

```text
{PIMoG, OLED, LED, Projector}
    x
{0.55, 0.70, 0.85, 1.00}
```

每个 cell 使用与 epoch 无关的固定 RNG stream，最多覆盖 16 个 validation batches。`use_fixed_matrix_for_checkpoint: true`，因此 checkpoint 比较使用固定矩阵的 macro ACC、worst-cell ACC 和 macro BCE，而不是某一次随机 mixed attack。

当前 `balanced_score_v1_fixed_matrix` 为：

```text
0.40 * degraded_macro_acc
+ 0.25 * degraded_worst_acc
+ 0.10 * clean_acc
+ 0.05 * exp(-degraded_macro_bce)
+ 0.10 * normalized_psnr(30 dB, 45 dB)
+ 0.10 * residual_quality(top-k, TV, channel balance)
```

其中 residual quality 是三个归一化分数的均值：`1-clamp(topk/0.10)`、`1-clamp(TV/0.05)` 与 `1-clamp(channel_std/0.02)`。

SSIM 和其他结构指标会记录到日志/checkpoint，但不直接进入该 score。训练会保存：

- `latest.pt`：按 `save_interval` 保存，用于恢复；
- `best_degradation_stage*.pt`：按当前候选退化集合分组的阶段最佳；
- `best.pt`：当前 degradation-stage 最佳的镜像；
- `final.pt`：最后一轮权重与最近一次 validation metadata。

正式报告仍应使用 `eval_watermark_robustness.py` 的完整多步嵌入重新比较候选 checkpoint，不能把 single-step validation 当作最终部署性能。

## 项目结构

```text
DiffuBind/
|-- configs/
|   |-- watermark_stage1.yaml       # clean Warmup/Balance/Full
|   `-- watermark_stage2.yaml       # seven-phase cross-device training
|-- dataset/
|   `-- watermark_image_dataset.py  # resize/crop and validation messages
|-- guided_diffusion/               # DDPM schedule and U-Net backbone
|-- models/
|   |-- watermark_unet.py           # carrier + global/spatial WM conditions
|   |-- watermark_residual.py       # content mask and residual projection
|   `-- watermark_decoder.py        # residual multi-scale Decoder
|-- NOISE_LAYER/
|   |-- build_noise_layer.py        # factory and MixedNoiseLayer
|   |-- PIMoG_Layer.py
|   |-- OLED_Layer.py
|   |-- LED_Layer.py
|   `-- Projector_Layer.py
|-- train_watermark_diffusion.py    # training, validation, checkpointing
|-- sample_embed_watermark.py       # single/batch full-DDPM embedding
|-- eval_watermark_robustness.py    # synthetic robustness evaluation
|-- eval_real_screen.py             # Decoder-only real capture evaluation
|-- manual_inspection.py            # manual four-corner rectification GUI
`-- PROJECT_MANUAL.md               # implementation audit notes
```

`models/screen_simulator.py` 是旧包装器，不属于当前正式训练主链；正式入口统一使用 `NOISE_LAYER/build_noise_layer.py`。

## 运行环境

建议使用独立的 Conda 或 Python 虚拟环境。下文命令均假设环境已经激活，并且 `python`、`pip` 已加入当前终端的 `PATH`，不依赖任何本机绝对路径或特定环境名称。

Windows 安装核心依赖：

```powershell
python -m pip install -r requirements.txt
```

Linux 安装核心依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 包含核心训练依赖。附加功能需要按用途安装：

- `manual_inspection.py`：`opencv-python`，以及系统可用的 Tk；
- `eval_watermark_robustness.py --enable_lpips`：可选 `lpips`；
- 部分 `tools/` 诊断脚本可能还需要 `matplotlib` 或 `tqdm`。

## 数据集

正式配置默认指向 COCO 2017 风格目录：

```text
/path/to/datasets/
|-- train2017/
`-- val2017/
```

在 `configs/watermark_stage1.yaml` 和 `configs/watermark_stage2.yaml` 中修改：

```yaml
data:
  train_dir: /path/to/train2017
  val_dir: /path/to/val2017
```

预处理保持长宽比：`Resize(shorter_edge=128)` 后，训练使用 `RandomCrop(128)`，验证使用 `CenterCrop(128)`，最后归一化到 `[-1,1]`。

数据发现逻辑并非完全递归：优先查找根目录的 PNG；没有 PNG 时查找根目录 JPG，再查找 JPEG，最后只回退到一层子目录 PNG。建议把同一数据集统一放在配置目录的当前层，避免扩展名优先级导致漏读。

## 训练命令

### 第一阶段

首次训练先把 `train.stage` 改为 `warmup`。

Windows：

```powershell
python train_watermark_diffusion.py --config configs\watermark_stage1.yaml
```

Linux：

```bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml
```

进入 Balance 或 Full 时，修改 `train.stage` 并从前一阶段初始化。

Windows：

```powershell
python train_watermark_diffusion.py --config configs\watermark_stage1.yaml --init_from checkpoints\stage1\best.pt
```

Linux：

```bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml \
  --init_from checkpoints/stage1/best.pt
```

同一阶段中断恢复。

Windows：

```powershell
python train_watermark_diffusion.py --config configs\watermark_stage1.yaml --resume checkpoints\stage1\latest.pt
```

Linux：

```bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml \
  --resume checkpoints/stage1/latest.pt
```

### 第二阶段

从 Stage 1 Full checkpoint 开始新的 Stage 2。

Windows：

```powershell
python train_watermark_diffusion.py --config configs\watermark_stage2.yaml --init_from checkpoints\stage1\best.pt
```

Linux：

```bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage2.yaml \
  --init_from checkpoints/stage1/best.pt
```

当前配置只要求新 Stage 2 提供某个 `--init_from`；`expected_init_from` 仍是注释，不会强制具体文件名。请自行确保它确实来自结构匹配的 Stage 1 Full。

恢复同一个 Stage 2。

Windows：

```powershell
python train_watermark_diffusion.py --config configs\watermark_stage2.yaml --resume checkpoints\stage2\latest.pt
```

Linux：

```bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage2.yaml \
  --resume checkpoints/stage2/latest.pt
```

`--resume` 与 `--init_from` 互斥：前者恢复模型、Decoder、optimizer、AMP scaler、step 与 RNG；后者只初始化模型/Decoder，并从新 optimizer 和 `global_step=0` 开始。

## 水印嵌入与采样

`sample_embed_watermark.py` 执行正式 image-to-image DDPM embedding，支持：

- 单张图片或目录当前层的非递归批处理；
- 随机水印或 `--watermark` 指定的二进制字符串；
- clean decoding；
- 指定单一/mixed degradation 后的 decoding；
- `--save_degraded` 导出固定 PIMoG/OLED/LED/Projector 版本；
- cover/watermarked comparison、degraded grid 和批量 CSV。

单图示例（Windows）：

```powershell
python sample_embed_watermark.py `
  --checkpoint checkpoints\stage2\best.pt `
  --config configs\watermark_stage2.yaml `
  --input test_images\test.png `
  --watermark "101100101011001010110010101100" `
  --output outputs\stage2\watermarked.png `
  --t_start 200
```

单图示例（Linux）：

```bash
python sample_embed_watermark.py \
  --checkpoint checkpoints/stage2/best.pt \
  --config configs/watermark_stage2.yaml \
  --input test_images/test.png \
  --watermark "101100101011001010110010101100" \
  --output outputs/stage2/watermarked.png \
  --t_start 200
```

目录批处理并导出固定退化（Windows）：

```powershell
python sample_embed_watermark.py `
  --checkpoint checkpoints\stage2\best.pt `
  --config configs\watermark_stage2.yaml `
  --input test_images `
  --output outputs\stage2\watermarked_batch `
  --t_start 200 `
  --noise_layer mixed `
  --save_degraded `
  --degradation_types pimog,oled,led,projector
```

目录批处理并导出固定退化（Linux）：

```bash
python sample_embed_watermark.py \
  --checkpoint checkpoints/stage2/best.pt \
  --config configs/watermark_stage2.yaml \
  --input test_images \
  --output outputs/stage2/watermarked_batch \
  --t_start 200 \
  --noise_layer mixed \
  --save_degraded \
  --degradation_types pimog,oled,led,projector
```

注意：

- CLI 默认 `--t_start 300`，但正式训练配置为 200；可比实验必须显式传入 `--t_start 200`；
- YAML 中的 `diffusion.sample_steps: 100` 当前没有被采样代码读取，不控制 reverse steps；
- 实际采样步数完全由 `--t_start` 决定；
- 指定消息不足 30 bit 时补 0，超过时截断；不指定消息时随机生成；
- 目录模式对 checkpoint、模型和 Decoder 只加载一次，但同一次运行中的所有图片共享同一组消息 bits。

## 合成退化鲁棒性评测

`eval_watermark_robustness.py` 先对固定验证子集执行完整 DDPM embedding，再分别应用 clean/PIMoG/OLED/LED/Projector/mixed 退化。

Windows：

```powershell
python eval_watermark_robustness.py `
  --checkpoint checkpoints\stage2\best.pt `
  --config configs\watermark_stage2.yaml `
  --noise_layers clean,pimog,oled,led,projector `
  --t_start 200 `
  --batch_size 8 `
  --seed 42 `
  --num_eval_images 500 `
  --subset_seed 42 `
  --num_visual_samples 16 `
  --noise_strength 1.0 `
  --attack_repeats 1 `
  --output outputs\stage2\eval_results_500.csv
```

Linux：

```bash
python eval_watermark_robustness.py \
  --checkpoint checkpoints/stage2/best.pt \
  --config configs/watermark_stage2.yaml \
  --noise_layers clean,pimog,oled,led,projector \
  --t_start 200 \
  --batch_size 8 \
  --seed 42 \
  --num_eval_images 500 \
  --subset_seed 42 \
  --num_visual_samples 16 \
  --noise_strength 1.0 \
  --attack_repeats 1 \
  --output outputs/stage2/eval_results_500.csv
```

支持的指标包括：

- Bit Accuracy、BER、30-bit Message Success Rate；
- cover -> watermarked：PSNR、SSIM、L1、optional LPIPS；
- watermarked -> degraded：attack PSNR、SSIM、L1；
- cover -> degraded：end-to-end PSNR、SSIM、L1；
- per-device aggregate、percentile/minimum 与 per-image rows。

输出包括 summary CSV、`*_by_noise.csv`、`*_per_image.csv`、metadata JSON、固定 subset indices 和 comparison images。`attack_repeats` 只重复相对便宜的退化与 Decoder，不重复 DDPM embedding。`num_eval_images=0` 表示完整验证集。

## 真实屏摄评测

### 人工透视矫正

`manual_inspection.py` 是 Tk/OpenCV GUI。它递归收集 `real_screen_photos/` 下的图片，允许在原始照片上手工选择显示区域四角，执行 perspective rectification，并把结果保存到 `real_screen_photos/rectified/`。该脚本只做人工检查与几何矫正，不做水印解码。

Windows：

```powershell
python manual_inspection.py
```

Linux：

```bash
python manual_inspection.py
```

### 仅使用解码器的真实屏摄评测

`eval_real_screen.py` 只加载 checkpoint 中的 Decoder：

```text
real captured image
    -> aspect-ratio-preserving resize
    -> center crop
    -> normalize to [-1,1]
    -> Decoder
    -> sigmoid(logits) > 0.5
    -> recovered bits
```

它不会重新运行 diffusion embedding。预先生成并显示水印图、完成真实拍摄和必要的透视矫正后，再执行。

Windows：

```powershell
python eval_real_screen.py `
  --checkpoint checkpoints\stage2\best.pt `
  --input_dir real_screen_photos\rectified `
  --watermark "101100101011001010110010101100" `
  --device cuda
```

Linux：

```bash
python eval_real_screen.py \
  --checkpoint checkpoints/stage2/best.pt \
  --input_dir real_screen_photos/rectified \
  --watermark "101100101011001010110010101100" \
  --device cuda
```

提供 expected watermark 时，脚本输出逐图 ACC/BER、总体平均，并写入输入目录下的 `real_screen_results.csv`；不提供时只输出 recovered bits。真实照片不是 synthetic degradation，二者的结果必须分开报告。

## 重要配置说明

| 配置项 | 当前真实效力 |
|---|---|
| `data.watermark_length` | 正式配置均为 30 |
| `data.image_size` | 正式配置均为 128 |
| `train.stage` | Stage 1 当前提交值为 `full`；Stage 2 的 stage 字符串不驱动嵌套阶段逻辑 |
| `train_watermark_mode` | Dataset 内有效，但训练 loop 会覆盖 `batch['wm_bits']`，不控制实际训练消息 |
| `val_watermark_mode` | 有效；validation 使用 Dataset 返回的固定 bits |
| `wm_t_min/max` | 控制训练/validation 的低时间步单步 watermark branch |
| `train_t_start` | 控制训练脚本周期完整采样，当前为 200 |
| `sample_steps` | 当前无 Python 调用，不生效 |
| `freeze_watermark_map_mlp` | Stage 2 为 false，因此 spatial map MLP 可更新 |
| `MixedNoiseLayer` | 一次 forward 只选择一种 degradation |
| `multi_attack.attacks_per_batch` | 对同一 `pred_x0` 最多执行 4 种不同 degradation |
| curriculum `apply_prob` | 控制该 batch 是否启用 degraded 分支 |
| curriculum `strength` | 以 `source + strength * (full_degraded - source)` 线性混合 |
| fixed validation matrix | 当前为 4 devices × 4 strengths，参与 checkpoint score |

## 已知限制与工程说明

1. 仓库不含数据、checkpoint 和最终日志，无法从当前提交复现或验证最终性能数值。
2. `train_watermark_mode` 当前不会改变训练 loop 实际使用的 bits；训练每个 batch 直接调用 `generate_train_watermark()`。
3. `sample_steps` 未接入采样；sample/eval CLI 默认 `t_start=300`，与正式配置 `train_t_start=200` 不同。
4. Stage 1 Full 训练通过 `train.stages.full` 使用 `wm_map_flat_floor=.03` 和 residual budget `.003/.040, power=2.8`；独立 sample/eval 当前只读取 top-level `model/train`，会分别得到 `.05` 和 `.004/.040, power=2.2`。Stage 1 checkpoint 的独立推理因此存在配置解析不一致；Stage 2 没有该嵌套覆盖问题。
5. sample/eval 对 diffusion model 使用 `strict=False` 加载；出现 missing/unexpected/mismatched 提示时，不应把结果视为可靠实验。
6. 训练期 fixed matrix 基于 single-step `pred_x0`，不是完整 DDPM embedding；它适合稳定排序，但不能代替最终多步评估。
7. `requirements.txt` 未覆盖人工校正、可选 LPIPS 和部分诊断脚本的全部附加依赖。
8. Stage 2 YAML 顶部的命令示例仍引用一个已经停用的旧配置文件名；仓库中的正式路径是 `configs/watermark_stage2.yaml`。
9. `PROJECT_MANUAL.md` 和 `manual_inspection.py` 的部分中文注释存在历史编码痕迹，不影响本文所述 Python 主链逻辑。

## 实验结果

当前提交不提供可核验的实验表格。

```text
Experimental results will be added after the final evaluation.
```

最终报告建议至少包含：固定 subset 与 full validation 的 per-device ACC/BER/message success、clean 与 end-to-end image quality、best/latest/final checkpoint 对比，以及分设备、分距离、分视角的真实屏摄结果。所有结果应记录 checkpoint、配置、`t_start`、subset indices、attack seed/strength/repeats 和拍摄设备条件。

## 引用与致谢

本项目使用并改造了 guided-diffusion 风格的 U-Net/DDPM 实现，并参考了以下方向：

- WaDiff：*A Watermark-Conditioned Diffusion Model for IP Protection*；
- PIMoG: *An Effective Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking Network*；
- projector-camera forward modeling / DeProCams 思路。

正式公开或投稿前，请根据实际采用的上游代码补齐许可证、版权声明、准确文献条目与 BibTeX。本仓库当前尚未提供 DiffuBind 论文的最终 citation。
