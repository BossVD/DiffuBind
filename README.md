# WaDiff — 基于扩散模型的水印嵌入与屏摄鲁棒性实验

## 核心思路

```
cover image + watermark bits → diffusion model → watermarked image
                                  ↓
                          mixed 屏摄/投影退化层
                                  ↓
                          watermark decoder → recovered bits
```

扩散模型在载体图像条件和水印条件共同引导下，对载体图像进行内容保持的重绘，
并在重绘过程中嵌入水印信息。训练和采样阶段均采用 image-to-image 范式，
始终从 cover image 的加噪版本出发，从不使用纯噪声 N(0,I)。

## 项目结构

```
guided_diffusion/          # 精简版 guided-diffusion（UNet + 扩散过程）
dataset/                   # 数据集加载（支持 max_images 限制）
models/                    # 条件 U-Net、水印解码器
NOISE_LAYER/               # 统一退化层（PIMoG、OLED、LED、Projector、Mixed）
configs/                   # YAML 配置文件
train_watermark_diffusion.py    # 训练脚本
sample_embed_watermark.py       # 采样/水印嵌入
eval_watermark_robustness.py    # 鲁棒性评估
```

## 快速开始

### 1. 安装环境

```bash
conda create -n wadiff python=3.10 -y
conda activate wadiff
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 准备数据

COCO 2017 数据集目录结构：

```
/path/to/datasets/
  train2017/
    000000000009.jpg
    ...
  val2017/
    000000000139.jpg
    ...
```

---

## 当前正式训练流程：Stage 1 → Stage 2

项目现在只使用两个正式训练阶段：

~~~text
Stage 1（无退化）
  warmup → balance → full，三个子阶段手动切换
  学习clean水印嵌入、提取、严格纹理定位和图像质量
                    ↓
Stage 2（有退化）
  从Stage 1 full的latest.pt初始化
  学习PIMoG/OLED/LED/Projector屏摄鲁棒性
~~~

Stage 1.5已从正式流程和配置中移除。

---

## Stage 1：clean嵌入与严格纹理定位

正式配置：

~~~text
configs/watermark_stage1.yaml
~~~

Stage 1不使用任何屏摄退化：

~~~yaml
noise_layer:
  type: none
~~~

Encoder和Decoder都完整训练。训练目标包括：

- 从cover image和30 bit生成watermarked image；
- Decoder从clean水印图恢复30 bit；
- 保持watermarked image与cover的图像质量；
- Full和Stage 2使用1/3边缘尺度和3/5纹理尺度构造精细内容mask；
- 将主要残差能量集中到多个边缘/纹理区域；
- 抑制平坦区域残差、固定横向条纹和窄带频谱峰值。

关键严格纹理参数（top-level为full/Stage 2最终值，warmup和balance
通过`train.stages`自动使用更宽松的阶段预算）：

~~~yaml
model:
  use_content_gated_wm_map: true
  wm_map_flat_floor: 0.03

train:
  residual_constraint:
    enabled: true
    flat_max_abs_delta_01: 0.003
    texture_max_abs_delta_01: 0.040
    mask_power: 2.8

  region_guidance:
    mode: strict_multiscale
    loss_mode: energy_ratio
    edge_scales: [1, 3]
    texture_scales: [3, 5]
    edge_weight: 0.70
    texture_weight: 0.30
    target_mask_area: 0.24
    mask_temperature: 0.03
    start_inside_ratio: 0.65
    target_inside_ratio: 0.85
    start_max_outside_ratio: 0.35
    max_outside_ratio: 0.15
    ratio_warmup_steps: 5000
~~~

输出目录：

~~~text
checkpoints_stage1_strict_texture_30bit_fine_v2
outputs_stage1_strict_texture_30bit_fine_v2/samples
outputs_stage1_strict_texture_30bit_fine_v2/logs
~~~

### Stage 1的三个手动子阶段

当前子阶段由以下字段控制：

~~~yaml
train:
  stage: warmup
~~~

只允许使用warmup、balance或full。阶段名拼错时训练脚本会直接终止。阶段切换必须使用--init_from；同一阶段中断恢复才使用--resume。

#### 1. warmup：建立clean水印通信

warmup重点训练写入和读取水印：

- lambda_wm较高；
- 使用lambda_diff=0.01维持全时间步扩散数值稳定；
- 图像约束较轻；
- 精细位置mask始终生效，但残差预算放宽为flat=0.012、texture=0.060、mask_power=1.0；
- `wm_map_flat_floor=0.20`，避免随机初始化阶段的bit条件被过早压弱；
- 纹理定位loss使用较小权重；
- 频谱正则暂时关闭，避免阻碍初始通信收敛。

配置：

~~~yaml
train:
  stage: warmup
~~~

Linux：

~~~bash
export OMP_NUM_THREADS=8

python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml
~~~

Windows：

~~~powershell
& "D:\Anaconda_envs\envs\wadiff\python.exe" train_watermark_diffusion.py --config configs\watermark_stage1.yaml
~~~

建议在以下条件基本满足后进入balance：

- train bit_acc稳定高于0.90；
- val bit_acc_clean持续上升并明显高于随机值；
- loss_wm稳定下降；
- residual没有大面积泄漏到平坦区域。

#### 2. balance：恢复图像质量并加强纹理定位

编辑配置：

~~~yaml
train:
  stage: balance
~~~

balance会分三段调整loss：

| step | lambda_diff | lambda_img | lambda_wm | lambda_region |
|---:|---:|---:|---:|---:|
| 0–2999 | 0.01 | 0.1 | 20.0 | 0.20 |
| 3000–5499 | 0.05 | 0.5 | 12.0 | 0.35 |
| 5500+ | 0.1 | 1.0 | 8.0 | 0.50 |

balance自动把残差预算收紧为flat=0.007、texture=0.050、mask_power=1.6，
并使用`wm_map_flat_floor=0.10`；引导支持区域从Warmup的30%收紧到27%，
同时从step 2000开始逐渐启用轻量方向和频谱约束。

~~~bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml \
  --init_from checkpoints_stage1_strict_texture_30bit_fine_v2/best.pt
~~~

#### 3. full：Stage 1最终联合收敛

编辑配置：

~~~yaml
train:
  stage: full
~~~

full同时优化diffusion噪声预测、clean水印提取、图像保真、RGB残差平衡、纹理能量比例及方向和FFT峰值。
该阶段使用最终精细引导：edge_scales=[1,3]、texture_scales=[3,5]、
target_mask_area=0.24、mask_temperature=0.03，并自动恢复最终严格预算
flat=0.003、texture=0.040、mask_power=2.8和`wm_map_flat_floor=0.03`。

当前主要权重：

~~~yaml
lambda_diff: 0.3
lambda_img: 3.0
lambda_wm: 3.0
lambda_delta: 0.05
lambda_channel: 0.2
lambda_region: 0.5
~~~

~~~bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml \
  --init_from checkpoints_stage1_strict_texture_30bit_fine_v2/best.pt
~~~

Stage 1的full阶段完成后，应使用通过多步嵌入检查的checkpoint。当前实验在Full指标
收敛后提前结束，因此Stage 2初始化保护指定为`latest.pt`。

### Stage 1中断恢复

保持当前train.stage不变：

~~~bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage1.yaml \
  --resume checkpoints_stage1_strict_texture_30bit_fine_v2/latest.pt
~~~

不要用--resume切换warmup、balance和full；切换阶段时使用--init_from。

### Stage 1重点指标

| 指标 | 建议目标 |
|---|---:|
| clean ACC | ≥0.98 |
| mask_area_ratio | 0.25–0.45 |
| inside_energy_ratio | ≥0.80，最终接近0.85 |
| outside_energy_ratio | ≤0.20 |
| directional_ratio | 尽量低于2 |
| PSNR变化 | 相比高画质基线不应明显下降 |
| 残差图 | 平坦区接近无信号，无跨图重复横纹 |

---

## Stage 2：mixed屏摄鲁棒训练

正式配置：

~~~text
configs/watermark_stage2.yaml
~~~

Stage 2只能从当前选定的Stage 1 Full checkpoint开始：

~~~text
checkpoints_stage1_strict_texture_30bit_fine_v2/latest.pt
~~~

训练脚本会拒绝无--init_from的新Stage 2运行，也会拒绝不符合配置要求的初始化路径。

### Stage 2训练重点

- Decoder完整训练，学习从屏摄退化图中恢复30 bit；
- Encoder只部分解冻，学习率为1e-6；
- Decoder学习率为2e-5；
- watermark_map_mlp保持冻结；
- 同一watermarked image每批训练两个不同退化；
- degraded loss使用mean + worst-case；
- 与Stage 1 Full完全一致地保持24%精细纹理mask、0.003/0.040区域残差预算、
  `mask_power=2.8`和`wm_map_flat_floor=0.03`；
- 继续保持Stage 1 Full的纹理内外能量目标和频谱约束；
- 使用固定4种退化×4种强度矩阵选择checkpoint。

| 退化 | 主要模拟内容 |
|---|---|
| PIMoG | 透视、光照、摩尔纹和噪声 |
| OLED | 子像素、显示模糊、频闪、反射和颜色变化 |
| LED | 降采样、灯珠/像素网格、scanline、摩尔纹和透视 |
| Projector | gamma、亮度衰减、热点、环境光、模糊和透视 |

### 退化课程

| phase | step范围 | candidates | strength | apply_prob | lambda degraded |
|---:|---:|---|---:|---:|---:|
| 1 | 0–1999 | Projector, OLED | 0.25 | 0.30 | 0.50 |
| 2 | 2000–5999 | Projector, OLED, PIMoG | 0.40 | 0.50 | 1.00 |
| 3 | 6000–13999 | 四种退化 | 0.55 | 0.70 | 1.50 |
| 4 | 14000–21999 | 四种退化 | 0.70 | 0.80 | 1.25 |
| 5 | 22000–27999 | 四种退化 | 0.85 | 0.90 | 1.00 |
| 6 | 28000+ | 四种退化 | 1.00 | 0.90 | 1.00 |

Phase 1的degraded分支对Encoder执行detach，主要让Decoder先适应温和退化；Phase 2之后才允许退化水印损失小幅更新部分Encoder。

### 启动Stage 2

Linux：

~~~bash
export OMP_NUM_THREADS=8

python train_watermark_diffusion.py \
  --config configs/watermark_stage2.yaml \
  --init_from checkpoints_stage1_strict_texture_30bit_fine_v2/latest.pt
~~~

Windows：

~~~powershell
& "D:\Anaconda_envs\envs\wadiff\python.exe" train_watermark_diffusion.py --config configs\watermark_stage2.yaml --init_from checkpoints_stage1_strict_texture_30bit_fine_v2\latest.pt
~~~

输出目录：

~~~text
checkpoints_stage2_one_shot_relaxed_v1
outputs_stage2_one_shot_relaxed_v1/samples
outputs_stage2_one_shot_relaxed_v1/logs
~~~

### Stage 2中断恢复

~~~bash
python train_watermark_diffusion.py \
  --config configs/watermark_stage2.yaml \
  --resume checkpoints_stage2_one_shot_relaxed_v1/latest.pt
~~~

恢复时不要再传--init_from。

### Stage 2日志与checkpoint

- train_log.csv：loss、clean/degraded ACC、实际退化、强度、梯度和残差结构；
- val_log.csv：clean和逐退化ACC、PSNR、纹理定位、频谱与综合评分；
- val_fixed_matrix.csv：固定退化/强度矩阵；
- sample_log.csv：完整多步嵌入样本的ACC和视觉指标。

| checkpoint | 用途 |
|---|---|
| best.pt | 当前固定矩阵综合评分最佳 |
| best_degradation_stage*.pt | 相同退化集合内的阶段最佳 |
| latest.pt | 中断恢复 |
| final.pt | 最后一轮权重 |

最终报告应同时比较best.pt、latest.pt和final.pt，并使用真正的多步嵌入评估。

---

## 采样（生成带水印图）

```bash
# 基础采样（随机水印）
/root/miniconda3/envs/wadiff/bin/python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/best.pt \
  --input ./test_images/test.png \
  --output ./outputs_stage2_one_shot_relaxed_v1/watermarked.png \
  --t_start 200

# 指定水印内容
/root/miniconda3/envs/wadiff/bin/python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/best.pt \
  --input ./test_images/test.png \
  --watermark "101100101011001010110010101100" \
  --output ./outputs_stage2_one_shot_relaxed_v1/watermarked_fixed_bits.png \
  --t_start 200

# 同时保存固定退化版本，便于区分 mixed 训练后的不同屏摄退化
/root/miniconda3/envs/wadiff/bin/python sample_embed_watermark.py \
  --config configs/watermark_stage2.yaml \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/best.pt \
  --input ./test_images/test.png \
  --output ./outputs_stage2_one_shot_relaxed_v1/watermarked_with_degradations.png \
  --t_start 200 \
  --noise_layer mixed \
  --save_degraded \
  --degradation_types pimog,oled,led,projector
```

当前 Stage 2 的水印长度为 30 bit。输入位数不足时会自动补 0，超出时会按照 checkpoint
配置截断。
采样始终输出 `bit_acc_clean`。只有指定了实际噪声层时才计算
`bit_acc_degraded`；默认 `--noise_layer none` 时日志显示
`bit_acc_degraded=N/A (not evaluated)`，避免把同一张 clean 水印图的重复解码
误认为退化鲁棒性。传入 `--save_degraded` 时，会把退化图、
cover/watermarked/degraded/residual grid，以及固定退化版本保存到输出目录下的
`degraded/` 子目录。

---

## 真实世界屏摄实验（Linux/AutoDL）

下面的命令均在项目根目录下执行，可以直接复制运行。实验使用固定的 30-bit
水印 `101100101011001010110010101100`，生成和解码时必须保持一致。

### 1. 创建实验目录

```bash
mkdir -p ./outputs_stage2_one_shot_relaxed_v1/real_world
mkdir -p ./real_screen_photos
```

### 2. 生成带水印图片

#### 单张图片

```bash
/root/miniconda3/envs/wadiff/bin/python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/final.pt \
  --input ./test_images/test.png \
  --watermark "101100101011001010110010101100" \
  --output ./outputs_stage2_one_shot_relaxed_v1/real_world/watermarked.png \
  --t_start 200 \
  --device cuda
```

命令执行完成后，用于真实屏摄的图片为：

```text
outputs_stage2_one_shot_relaxed_v1/real_world/watermarked.png
```

将这张图片全屏显示在显示器、OLED、LED 或投影设备上，然后使用手机或相机拍摄。
不要拍摄 `comparison/` 目录中的对比图，也不要使用 `degraded/` 目录中的模拟退化图。

#### 目录中的全部图片（非递归）

如果 `test_images/` 中有多张图片，可以直接把目录传给 `--input`。`--output`
此时必须是输出目录：

```bash
/root/miniconda3/envs/wadiff/bin/python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/final.pt \
  --input ./test_images \
  --watermark "101100101011001010110010101100" \
  --output ./outputs_stage2_one_shot_relaxed_v1/real_world/watermarked_batch \
  --t_start 200 \
  --device cuda
```

该命令只处理 `test_images/` 当前层中的 `.jpg`、`.jpeg`、`.png` 和 `.bmp`
文件，不进入子目录。checkpoint、扩散模型和 decoder 只加载一次，所有图片使用
同一个固定水印并依次处理，不会一次性把全部图片放入显存。

假设输入目录为：

```text
test_images/
├── image001.jpg
├── image002.png
└── image003.bmp
```

主要输出为：

```text
outputs_stage2_one_shot_relaxed_v1/real_world/watermarked_batch/
├── image001_watermarked.png
├── image002_watermarked.png
├── image003_watermarked.png
├── batch_embed_results.csv
└── comparison/
    ├── image001_watermarked_comparison.png
    ├── image002_watermarked_comparison.png
    └── image003_watermarked_comparison.png
```

输出统一使用无损 PNG。`batch_embed_results.csv` 保存每张图片的输入路径、输出
路径、clean/degraded bit accuracy、处理状态和错误信息。当
`--noise_layer none` 时，CSV 的 `bit_acc_degraded` 单元格留空，表示未执行退化
评估；只有实际指定噪声层时该列才写入数值。如果某张图片读取失败，脚本会记录
错误并继续处理剩余图片，最后返回非零退出码以提示批次中存在失败项。

输入输出参数兼容以下形式：

| `--input` | `--output` | 行为 |
|---|---|---|
| 图片文件 | 图片文件 | 使用指定输出文件名，保持原有单图行为 |
| 图片文件 | 目录 | 自动保存为 `<原文件名>_watermarked.png` |
| 目录 | 目录 | 非递归处理目录当前层中的所有支持图片 |
| 目录 | 图片文件 | 报错，避免多张结果覆盖同一个文件 |

目录批量模式下，输入目录和输出目录不能相同。

### 3. 准备真实拍摄照片

将真实拍摄的图片上传到项目的 `real_screen_photos/` 目录，例如：

```text
real_screen_photos/
├── monitor_front_01.jpg
├── monitor_angle_01.jpg
├── oled_far_01.jpg
├── led_lowlight_01.jpg
└── projector_angle_01.jpg
```

支持 `.jpg`、`.jpeg`、`.png` 和 `.bmp`。运行解码前，建议先完成以下处理：

- 裁掉屏幕边框、桌面 UI、黑边和周围环境；
- 斜拍照片根据显示区域四角进行透视校正；
- 只保留屏幕中显示的水印图片区域；
- 将图片保存为正方形 RGB 图像。

脚本会自动缩放和中心裁剪，不需要手动调整到 `128×128`，但不会自动检测屏幕
边界或进行透视校正。

### 4. 解码并计算真实屏摄准确率

```bash
/root/miniconda3/envs/wadiff/bin/python eval_real_screen.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/final.pt \
  --input_dir ./real_screen_photos \
  --watermark "101100101011001010110010101100" \
  --device cuda
```

脚本会输出每张照片的解码结果、bit accuracy 和所有照片的平均准确率，并将结果
保存到：

```text
real_screen_photos/real_screen_results.csv
```

如果只想输出解码后的水印、不计算准确率，直接运行：

```bash
/root/miniconda3/envs/wadiff/bin/python eval_real_screen.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/final.pt \
  --input_dir ./real_screen_photos \
  --device cuda
```

同一目录中的照片应使用同一个固定水印。如果不同图片使用了不同水印，需要按照
水印内容分别放入不同目录并分别执行解码命令。

### 5. 使用其他 checkpoint 对比

例如，使用四种退化全部加入后的综合最佳 checkpoint 重新生成图片：

```bash
/root/miniconda3/envs/wadiff/bin/python sample_embed_watermark.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/best_degradation_stage3.pt \
  --input ./test_images/test.png \
  --watermark "101100101011001010110010101100" \
  --output ./outputs_stage2_one_shot_relaxed_v1/real_world/watermarked_best_stage3.png \
  --t_start 200 \
  --device cuda
```

完成显示、拍摄和照片裁剪后，将对应照片放入单独目录：

```bash
mkdir -p ./real_screen_photos_best_stage3
```

然后直接运行：

```bash
/root/miniconda3/envs/wadiff/bin/python eval_real_screen.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/best_degradation_stage3.pt \
  --input_dir ./real_screen_photos_best_stage3 \
  --watermark "101100101011001010110010101100" \
  --device cuda
```

比较不同 checkpoint 时，每个 checkpoint 都应分别生成带水印图片、进行屏摄并
使用对应 decoder 解码。仅更换 checkpoint 解码同一张照片，不能代表完整系统性能。

---

## 测试噪声层

```bash
/root/miniconda3/envs/wadiff/bin/python tools/test_noise_layer.py \
  --input ./tools/test.jpg \
  --config configs/watermark_stage2.yaml \
  --image_size 128 \
  --device cuda
```

---

## 屏摄鲁棒性评估

评估脚本加载已有 checkpoint，在固定验证集上执行完整 DDPM 水印采样，
不会修改模型权重，也不需要重新训练。扩散模型和 decoder 使用 `[-1,1]`，
退化层以及 PSNR、SSIM、L1 使用 `[0,1]`。

### 快速固定子集评估

下面的命令从验证集中固定抽取 500 张图。相同的 `subset_seed` 会得到相同
验证索引，适合比较 `best.pt`、`latest.pt` 和 `final.pt`：

```bash
/root/miniconda3/envs/wadiff/bin/python eval_watermark_robustness.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/final.pt \
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
  --output ./outputs_stage2_one_shot_relaxed_v1/eval_results_500.csv
```

`t_start=200` 表示每个 batch 执行完整的 200 次 DDPM 反向更新，不是采样
200 张图片。500 张、batch size 为 8 时约有 63 个 batch；完整 5000 张则
约有 625 个 batch。

快速评估可以暂时不加入 `mixed`。`mixed` 不是第五种物理退化，而是根据
`noise_layer.mixed.probs` 从 PIMoG、OLED、LED、Projector 中随机选择一种。
具体定位弱项时优先查看四种独立退化。

### 完整验证集评估

`num_eval_images=0` 表示使用完整验证集。最终报告可以增加随机攻击重复次数；
同一张水印图只做一次 200-step 采样，`attack_repeats` 只重复相对便宜的
退化层和 decoder：

```bash
/root/miniconda3/envs/wadiff/bin/python eval_watermark_robustness.py \
  --checkpoint checkpoints_stage2_one_shot_relaxed_v1/final.pt \
  --config configs/watermark_stage2.yaml \
  --noise_layers clean,pimog,oled,led,projector,mixed \
  --t_start 200 \
  --batch_size 8 \
  --seed 42 \
  --num_eval_images 0 \
  --subset_seed 42 \
  --num_visual_samples 16 \
  --noise_strength 1.0 \
  --attack_repeats 3 \
  --output ./outputs_stage2_one_shot_relaxed_v1/eval_results_full.csv
```

不要仅为了加速把 `t_start=200` 改成 50；这会改变采样任务，结果不能与
200-step 指标直接比较。显存允许时可以尝试提高 `batch_size`。

### 退化强度

`noise_strength` 使用与训练相同的线性混合定义：

```python
degraded_01 = watermarked_01 + noise_strength * (
    full_degraded_01 - watermarked_01
)
```

- `--noise_strength 1.0`：完整退化，适合最终鲁棒性评估；
- `--noise_strength 0.55/0.70/0.85`：与对应中间课程阶段对齐；
- `clean` 分支不施加退化，强度固定记为 0。

当前 Stage 2 最后一个课程阶段本身已经是 `strength=1.0`，因此 `final.pt` 的课程强度
与完整强度相同。评估中间阶段 checkpoint 时，建议同时保留对应课程强度与
`1.0` 完整强度结果，并在文件名中注明强度。

### 指标口径

脚本逐图计算并汇总：

| 类别 | 比较对象 | 指标 |
|------|------|------|
| 水印不可见性 | cover ↔ watermarked | PSNR、SSIM、L1，可选 LPIPS |
| 攻击强度 | watermarked ↔ degraded | PSNR、SSIM、L1 |
| 完整链路 | cover ↔ degraded | PSNR、SSIM、L1 |
| 水印提取 | target bits ↔ decoded bits | bit ACC、BER、30-bit message success |

每种退化输出 `bit_acc_mean`、`bit_acc_p5`、`bit_acc_min`、
`message_success_rate`。完整消息成功要求一张图的 30 bit 全部正确。

LPIPS 默认关闭，只有 WaDiff 环境已经安装可选的 `lpips` 包时才使用：

```bash
--enable_lpips
```

脚本不会自动安装依赖；未安装时不要传入该参数。

### 输出文件

以 `--output ./outputs_stage2_one_shot_relaxed_v1/eval_results_500.csv` 为例，会生成：

```text
outputs_stage2_one_shot_relaxed_v1/
├── eval_results_500.csv
├── eval_results_500_by_noise.csv
├── eval_results_500_per_image.csv
├── eval_results_500_metadata.json
├── eval_results_500_indices.txt
└── eval_results_500_samples/
    ├── 0000_comparison.png
    ├── 0001_comparison.png
    └── ...
```

- 主 CSV 保留 `metric,value` 格式；
- `by_noise.csv` 每行对应一种退化；
- `per_image.csv` 保存真正的逐图指标，不再复制 batch PSNR；
- metadata 保存 checkpoint、seed、强度、步数和实验状态；
- indices 保存实际使用的原验证集索引；
- 每个 comparison PNG 将同一张图的 cover、watermarked、各退化和
  `signed residual ×5` 放在一张图上。

对比图直接复用计算 ACC 时的退化 tensor，不会为了保存图片重新随机攻击。
默认只保存综合对比图；如需同时保存单独 PNG，追加：

```bash
--save_individual_samples
```

评估过程中会打印处理数量、batch、用时、ETA 和各退化的临时平均 ACC。
按 `Ctrl+C` 中断时，脚本会把已经完成的结果以 `status=interrupted` 写入
CSV 和 metadata。如果不传 `--data_dir`，验证目录使用配置中的
`data.val_dir`。

---

## 训练模式切换

| 模式 | max_train_images | epochs | image_size | 用途 |
|------|:---:|:---:|:---:|------|
| 快速调试 | 10000 | 10 | 64 | 验证流程跑通 |
| Stage 1 | 10000 | 80（每个手动子阶段上限） | 128 | clean嵌入、严格纹理定位与画质收敛 |
| Stage 2 | 10000 | 100 | 128 | 渐进式四退化鲁棒训练 |

只需改 YAML，不需要改代码。

## 关键设计

- **保持图像比例**：训练集将短边缩放到目标尺寸后随机裁剪；验证、采样和实拍评估使用中心裁剪，不把原图强制拉伸成正方形
- **确定性水印**：验证集根据相对路径和 `data.watermark_seed` 固定水印；训练集按图片和 epoch 可复现地变化，避免模型记忆“图片→水印”映射
- **实验随机种子**：`train.seed` 统一控制 Python、NumPy、PyTorch 和 DataLoader，检查点同时保存随机状态以支持可复现恢复
- **显存控制**：`use_amp: true` 启用真实 autocast；扩散分支先反向并释放计算图，再运行水印分支，避免同时保留两套 U-Net 激活
- **课程对齐验证**：Validation 使用当前 curriculum 的候选类型和强度、逐类型统计 ACC/BCE，并固定 bits 与 RNG；`apply_prob` 仅控制训练，验证始终施加攻击
- **最佳模型指标**：每个验证 epoch 计算 macro/worst ACC、clean ACC、BCE、PSNR 和残差质量组成的 `balanced_score_v1`，不再只按 degraded ACC 保存
- **按退化集合保存**：只有候选退化集合发生变化时才建立新的 `best_degradation_stage*.pt`；强度变化不会新增 best 文件
- **梯度流**：Stage 1完整训练Encoder和Decoder；Stage 2首个课程阶段阻断degraded分支到Encoder的梯度，后续仅小幅更新部分Encoder
- **水印解码器**：默认使用 residual multi-scale decoder；如需消融旧版 CNN baseline，可在 YAML 中设置 `decoder.type: simple`
- **t_diff / t_wm 分离**：噪声预测用全时间步，水印损失用小时间步保证 pred_x0 稳定
- **单步训练与完整采样**：TRAIN/VALIDATE 使用单步 `pred_x0`；周期 SAMPLE 和正式嵌入使用完整 DDPM 反向采样，两者指标不能直接横向等同
- **图像范围**：扩散模型和 decoder 使用 `[-1,1]`；统一退化层输入输出使用 `[0,1]`，训练接入点负责转换
- **image-to-image**：训练和采样始终从 cover 加噪出发，非纯噪声生成
- **统一退化配置**：仅通过 `noise_layer.type` 选择退化层，支持 `none`、`pimog`、`oled`、`led`、`projector` 和 `mixed`

## 参考文献

- WaDiff (ECCV 2024): [A Watermark-Conditioned Diffusion Model for IP Protection](https://arxiv.org/abs/2403.10893)
- PIMoG (MM 2022): [An Effective Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking Network](https://doi.org/10.1145/3503161.3548049)
