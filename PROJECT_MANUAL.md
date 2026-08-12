# CD-SCR-DiffWatermark 技术手册

> 审计基线：2026-08-13 当前工作区。本手册以现存 Python 源码和 `configs/watermark_stage1.yaml`、`configs/watermark_stage2.yaml` 为准。仓库不包含训练数据与 checkpoint，因此不对模型精度作未经复现的承诺。

## 1. 系统边界

本项目实现一个以载体图像和二进制消息为联合条件的扩散水印系统。核心链路为：

```text
cover [-1,1] + watermark bits [0,1]
  → cover 前向加噪得到 x_t
  → [x_t, cover, watermark spatial map] 输入条件 U-Net
  → U-Net 预测 epsilon
  → 解析恢复 pred_x0
  → 内容相关的残差预算投影
  → clean Decoder
  → 可选 PIMoG/OLED/LED/Projector 退化
  → degraded Decoder
  → BCEWithLogitsLoss / bit metrics
```

训练和采样均属于 image-to-image：反向过程从 cover 的加噪版本开始，而不是从纯噪声开始。扩散模型与 Decoder 的图像范围为 `[-1,1]`，统一退化层的输入输出范围为 `[0,1]`。

当前正式任务固定为 `128×128` RGB 图像和 30-bit 水印。两份 YAML 均使用 1000-step 线性 beta 调度、水印训练时间步 `[0,200)` 和完整嵌入起点 `train_t_start=200`。

## 2. 当前仓库结构

```text
CD-SCR-DiffWatermark/
├── configs/
│   ├── watermark_stage1.yaml          # clean 三子阶段训练
│   └── watermark_stage2.yaml          # one-shot mixed 鲁棒训练
├── dataset/
│   └── watermark_image_dataset.py     # 图像预处理与确定性验证 bits
├── guided_diffusion/
│   ├── gaussian_diffusion.py          # q/p 调度与 DDPM/DDIM 基础函数
│   ├── unet.py                        # U-Net 主干
│   ├── nn.py
│   ├── losses.py
│   ├── fp16_util.py
│   └── logger.py
├── models/
│   ├── watermark_unet.py              # cover + bits 条件包装器
│   ├── watermark_decoder.py           # 多尺度消息解码器
│   ├── watermark_residual.py          # 内容 mask 与残差约束
│   └── screen_simulator.py            # 未被正式主链采用的旧包装器
├── NOISE_LAYER/
│   ├── build_noise_layer.py           # 工厂与 MixedNoiseLayer
│   ├── PIMoG_Layer.py
│   ├── OLED_Layer.py
│   ├── LED_Layer.py
│   ├── Projector_Layer.py
│   └── utils.py
├── tools/                              # 静态/数值测试与退化诊断
├── train_watermark_diffusion.py       # 训练、验证、周期采样、checkpoint
├── sample_embed_watermark.py          # 单图/目录 DDPM 嵌入
├── eval_watermark_robustness.py       # 固定验证集合成退化评估
├── eval_real_screen.py                # 真实屏摄照片 Decoder 评估
└── manual_inspection.py               # 手动四点透视校正 GUI
```

`docs/`、`figures/`、数据、checkpoint、训练输出和真实屏摄照片均为本地材料，已由 `.gitignore` 排除。

## 3. 数据与消息

### 3.1 图像发现和预处理

`WatermarkImageDataset` 从给定目录收集图像，支持 JPG、JPEG 和 PNG。预处理流程为：

```text
PIL RGB
  → Resize(image_size)，保持长宽比
  → train: RandomCrop / val: CenterCrop
  → ToTensor [0,1]
  → Normalize(mean=.5, std=.5)
  → tensor [-1,1]
```

正式配置的数据路径当前是：

```yaml
train_dir: /root/autodl-tmp/datasets/train2017
val_dir: /root/autodl-tmp/datasets/val2017
```

这些只是 AutoDL/Linux 示例。任何新环境运行前都必须修改两份配置。

### 3.2 消息来源

Dataset 支持 fixed、deterministic_random、per_epoch、random 等模式，并可根据相对路径、seed 和 epoch 构造可复现 bits。

当前实际训练循环在每个 batch 内执行：

```python
wm_bits = generate_train_watermark(B, watermark_length, device)
```

因此训练时 Dataset 返回的 `batch['wm_bits']` 会被覆盖，`train_watermark_mode` 不控制送入 U-Net 和 Decoder loss 的训练消息。验证代码使用 `v_batch['wm_bits']`，所以 `val_watermark_mode` 有效。

## 4. 条件扩散模型

### 4.1 水印条件的两条路径

`WatermarkConditionedUNet` 将 bits 同时编码为：

1. 全局向量：水印 MLP 生成 `cond_dim=256` embedding，与时间 embedding 相加。
2. 空间条件：另一 MLP 生成 `4×16×16` map，再双线性插值至 `4×128×128`。

启用内容门控时，空间 map 会乘以由 cover 计算的边缘/纹理 allowance。最终 U-Net 输入为：

```text
x_t       [B,3,H,W]
cover     [B,3,H,W]
wm_map    [B,4,H,W]
concat →  [B,10,H,W]
```

当前 `base_channels=64`、`cond_dim=256`。代码依赖 `cond_dim == 4 × base_channels` 才能直接将水印 embedding 与时间 embedding 相加，配置层没有单独的早期断言。

### 4.2 扩散目标

前向加噪：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
```

模型预测 `epsilon`，再解析恢复：

```text
pred_x0 = x_t / sqrt(alpha_bar_t)
          - sqrt(1/alpha_bar_t - 1) * epsilon_pred
```

训练包含两条 U-Net 分支：

- 全时间步分支：在 `[0,999]` 采样 t，优化噪声预测 MSE。
- 水印分支：在 `[wm_t_min,wm_t_max)` 采样较低 t，恢复 `pred_x0` 后优化水印与视觉目标。

扩散分支先反向并释放计算图，随后才执行水印分支，以控制显存占用。

### 4.3 内容相关残差投影

`models/watermark_residual.py` 根据 cover 的多尺度边缘与纹理响应构造软 mask。原始残差在 `[0,1]` 域中经过空间预算限制：

```text
budget = mask^p * texture_budget + (1-mask^p) * flat_budget
bounded_delta = budget * tanh(raw_delta / budget)
watermarked = cover + bounded_delta
```

这意味着系统不是简单叠加固定扰动；扩散模型先预测候选图像，再将其投影到内容相关的残差预算内。

## 5. Decoder

正式配置使用 `residual_multiscale` Decoder：

```text
RGB image
  → stem Conv/GN/SiLU
  → 四个下采样残差 stage
  → 32×32、16×16、8×8 多尺度全局池化
  → concat
  → Linear/SiLU/Dropout
  → 30 logits
```

Decoder 不在内部执行 sigmoid。训练使用 `BCEWithLogitsLoss`；推理时以 `sigmoid(logits) > 0.5` 得到 bits。

仓库仍保留 `simple` Decoder 作为消融兼容选项，但现存两份正式 YAML 都使用 `residual_multiscale`。

## 6. 屏摄与投影退化

正式主链通过 `NOISE_LAYER.build_noise_layer()` 构建以下退化：

| 类型 | 主要近似因素 |
|---|---|
| PIMoG | 透视、光照、摩尔纹、模糊与噪声 |
| OLED | tone、PenTile/stripe 子像素、频带、视角色偏、反射、相机模糊与噪声 |
| LED | 低分辨率、灯珠/像素网格、scanline、摩尔纹、透视与相机链 |
| Projector | gamma、亮度衰减、热点、表面纹理、环境光、模糊、透视与颜色变化 |
| mixed | 从上述物理退化候选中选择，不是独立的第五种物理模型 |

退化层在 `[0,1]` FP32 中运行，随后转换回 `[-1,1]` 供 Decoder 使用。

## 7. Stage 1

配置：`configs/watermark_stage1.yaml`。

### 7.1 初始化约束

Stage 1 由 `train.stage` 手动选择 `warmup`、`balance` 或 `full`：

- Warmup 可从零开始。
- Balance 和 Full 必须使用 `--init_from` 或继续同阶段的 `--resume`。
- 阶段名不合法时训练脚本终止。
- `--resume` 与 `--init_from` 互斥。

当前 YAML 保存为 `stage: full`，因此直接不带 checkpoint 启动会被拒绝。首次训练必须先改为 `warmup`。

### 7.2 三个子阶段

| 阶段 | `wm_map_flat_floor` | flat/texture budget | `mask_power` | 主要目的 |
|---|---:|---:|---:|---|
| Warmup | 0.20 | 0.012 / 0.060 | 1.0 | 建立 clean 通信，频谱约束关闭 |
| Balance | 0.10 | 0.007 / 0.050 | 1.6 | 分段恢复图像质量并收紧空间支持 |
| Full | 0.03 | 0.003 / 0.040 | 2.8 | 最终严格视觉预算和频谱控制 |

Full 的主要 loss 权重：

```text
diff=.3, image=3, watermark=3, delta=.05, channel=.2, region=.5
```

Stage 1 使用 `noise_layer.type=none`，Encoder 和 Decoder 均完整训练。

### 7.3 输出

```text
checkpoints_stage1_strict_texture_30bit_fine_v2/
outputs_stage1_strict_texture_30bit_fine_v2/samples/
outputs_stage1_strict_texture_30bit_fine_v2/logs/
```

## 8. Stage 2

配置：`configs/watermark_stage2.yaml`。

### 8.1 初始化与优化器

Stage 2 设置 `require_init_from_on_new_run: true`，新实验必须从 Stage 1 Full checkpoint 初始化。当前 `expected_init_from` 仅保留为注释，不会强制某个具体文件名。

当前参数更新策略：

| 项目 | 设置 |
|---|---|
| Encoder 模式 | partial |
| 解冻 U-Net 输出块 | 最后 3 个 |
| `watermark_map_mlp` | 不冻结 |
| Encoder LR | `1e-6`，再乘课程 `lr_scale` |
| Decoder LR | `2e-5` |
| Epoch | 100 |
| AMP | 开启 |
| 梯度裁剪 | 1.0 |

`--init_from` 只加载模型与 Decoder 权重，并建立新的 Stage 2 AdamW；`--resume` 恢复同一 Stage 2 实验的训练状态。

### 8.2 视觉与多攻击目标

Stage 2 使用比 Stage 1 Full 略宽松的载体：

```text
wm_map_flat_floor = 0.07
flat/texture budget = 0.0035/0.050
mask_power = 2.1
target_mask_area = 0.31
target_inside_ratio = 0.80
max_outside_ratio = 0.20
```

同一 `pred_x0` 最多生成 4 个攻击结果。degraded BCE 聚合为：

```text
0.30 × mean attack loss + 0.70 × worst attack loss
```

### 8.3 七段退化课程

| step | candidates | apply | strength | clean/degraded 权重 | detach Encoder | LR scale |
|---:|---|---:|---:|---:|---|---:|
| `<2000` | Projector, OLED | .30 | .25 | 1.0 / .5 | 是 | .5 |
| `<6000` | Projector, OLED, PIMoG | .50 | .40 | .9 / 1.0 | 否 | .75 |
| `<12000` | 四种 | .70 | .60 | .75 / 1.5 | 否 | 1.0 |
| `<18000` | 四种 | .85 | .75 | .60 / 1.5 | 否 | 1.0 |
| `<24000` | 四种 | 1.0 | .90 | .50 / 1.75 | 否 | 1.0 |
| `<28000` | 四种 | 1.0 | 1.0 | .40 / 2.0 | 否 | 1.0 |
| 之后 | 四种 | 1.0 | .95 | .40 / 1.75 | 否 | .5 |

第一段只让 degraded 分支训练 Decoder；clean 分支仍可训练 Encoder。第二段开始 degraded 梯度可穿过退化层返回已解冻的 Encoder 参数。

### 8.4 验证与 checkpoint

每个验证 epoch 可同时计算课程视图和固定矩阵。固定矩阵为：

```text
PIMoG / OLED / LED / Projector
× strength 0.55 / 0.70 / 0.85 / 1.00
× 最多 16 个 validation batches
```

`use_fixed_matrix_for_checkpoint=true`，因此固定矩阵综合评分参与 checkpoint 选择。主要文件：

```text
latest.pt
best.pt
best_degradation_stage*.pt
final.pt
```

日志：

```text
train_log.csv
val_log.csv
val_fixed_matrix.csv
sample_log.csv
```

输出根目录：

```text
checkpoints_stage2_one_shot_relaxed_v1/
outputs_stage2_one_shot_relaxed_v1/
```

## 9. 一次训练 batch 的调用链

```text
Dataset cover
  → 训练循环重新生成随机 30 bits
  → 计算 detached 内容 allowance
  → 全时间步 q_sample + U-Net + epsilon MSE backward
  → 低时间步 q_sample + U-Net + pred_x0
  → 内容残差投影
  → clean Decoder BCE
  → Stage 2: 根据 curriculum 生成最多 4 个 degraded tensors
  → degraded Decoder mean/worst BCE
  → image/channel/region/spectral objectives
  → AMP unscale
  → finite gradient guard
  → gradient clip
  → optimizer step
  → CSV/终端日志
```

## 10. 完整嵌入与评估

### 10.1 `sample_embed_watermark.py`

支持：

- 单个输入文件或目录当前层批处理；
- 随机 bits 或指定二进制字符串；
- clean 解码；
- 指定 mixed/单一退化的 degraded 解码；
- 保存四种固定退化版本和 comparison grid。

输入为目录时输出也必须是目录，且输入输出目录不能相同。单张失败不会阻断其余批次，但批次最终返回非零退出码。

CLI 默认 `t_start=300`，正式配置为 200。可比实验应显式传 `--t_start 200`。

### 10.2 `eval_watermark_robustness.py`

脚本只读 checkpoint，在固定验证子集上完成多步 DDPM 嵌入，再计算：

- cover ↔ watermarked：PSNR、SSIM、L1、可选 LPIPS；
- watermarked ↔ degraded：退化强度指标；
- cover ↔ degraded：端到端视觉指标；
- target ↔ decoded：bit ACC、BER、30-bit message success。

`num_eval_images=0` 表示完整验证集。`attack_repeats` 只重复相对便宜的退化与 Decoder，不重复 DDPM 嵌入。中断时脚本会保存已完成结果并标记 `status=interrupted`。

### 10.3 `eval_real_screen.py`

该脚本只构建 Decoder，不运行扩散模型。它读取输入目录当前层的图片，保持比例缩放后中心裁剪，输出预测 bits；提供期望 bits 时，同时计算逐图和平均 ACC/BER，并写入 `real_screen_results.csv`。

真实斜拍图应先裁除背景并完成透视校正。`manual_inspection.py` 可通过 Tk GUI 手动选择屏幕四角并用 OpenCV 矫正。

## 11. Checkpoint 内容与加载语义

训练 checkpoint 保存模型、Decoder、优化器、AMP scaler、epoch、global step、配置、随机状态和验证元数据。恢复语义：

- `--resume`：严格继续同一阶段，恢复训练状态。
- `--init_from`：开始新阶段，加载模型/Decoder，重新建立优化器。

训练恢复路径使用严格模型加载。独立采样与鲁棒性评估当前对扩散模型使用 `strict=False`；Decoder 使用兼容加载并打印 missing、unexpected、mismatched 信息。出现这些提示时不能把结果当作可靠实验。

## 12. 配置项的实际效力

| 配置项 | 当前状态 |
|---|---|
| `train_watermark_mode` | Dataset 内有效，但训练 loop 会覆盖 bits，故不控制训练目标 |
| `val_watermark_mode` | 有效 |
| `wm_t_min/max` | 控制训练/验证低时间步分支 |
| `train_t_start` | 控制训练中的周期完整嵌入 |
| `sample_steps` | 当前没有 Python 调用，不控制采样步数 |
| `noise_layer.*.p` | 单退化/选择后应用概率有效；正式 Stage 2 各具体层为 1.0 |
| curriculum `apply_prob` | 控制是否运行 degraded 分支 |
| curriculum `probs` | 控制候选未全部选满时的采样权重 |
| `multi_attack.attacks_per_batch` | 当前为 4 |
| `validation.fixed_*` | 控制固定退化强度矩阵 |

## 13. 活跃代码与遗留代码

活跃主链：

- `GaussianDiffusion.q_sample()`、epsilon→x0、DDPM posterior；
- `WatermarkConditionedUNet` 手工驱动 guided-diffusion U-Net 主干；
- `watermark_residual.py` 的内容 mask 与残差约束；
- residual multi-scale Decoder；
- `NOISE_LAYER` 工厂和四种退化。

保留但不是当前正式训练主链：

- `guided_diffusion.gaussian_diffusion.training_losses()` 中的旧训练分支；
- `UNetModel.secret_dense` 的旧水印路径，包装器以 `wm_length=0` 禁用；
- `guided_diffusion/fp16_util.py` 的旧混合精度训练器，正式代码使用 `torch.amp`；
- `models/screen_simulator.py` 的旧 PIMoG 包装；
- `simple` Decoder；
- TensorBoard 依赖，当前训练日志实际写 CSV、PNG 和终端。

## 14. 已知风险与上传前说明

1. `sample_steps` 是未生效配置；正式命令必须显式写 `--t_start`。
2. sample/eval CLI 默认 `t_start=300`，与当前训练配置 200 不一致。
3. 训练 bits 会覆盖 Dataset bits，README 或论文不能将 `train_watermark_mode` 描述为当前训练消息策略。
4. 独立采样和评估的扩散模型加载不是 strict，需检查终端兼容性提示。
5. `tools/test_strict_texture_stage.py` 仍引用旧的 `configs/watermark_stage2_mixed_strict_texture_v1.yaml`，因此其中 two-stage config 测试会在读取路径时失败；这不是数值模型测试失败。
6. `requirements.txt` 未包含 `opencv-python`、`matplotlib`、`tqdm` 和可选 `lpips`；核心训练依赖已列出，附加工具需按用途安装。
7. 当前仓库无数据和 checkpoint，无法在本地复现最终 ACC、BER、PSNR 或真实屏摄结果。
8. `guided_diffusion` 和 WaDiff 派生代码公开分发前需补齐适用的第三方许可证、版权与引用说明。

## 15. 验证记录

本次文档更新执行的是只读/静态验证：

- 35 个 Python 文件完成 AST 解析；
- 2 份 YAML 使用 PyYAML 成功解析；
- 未发现 API key、访问令牌或密码样式的敏感内容；
- 最大本地文件约 1.07 MB，无 GitHub 单文件体积风险；
- 数据、checkpoint、outputs、`docs/`、`figures/`、真实屏摄照片和本地调试图片均由 `.gitignore` 排除。

没有启动训练，也没有对模型、loss、网络结构或超参数作修改。

## 16. 推荐阅读顺序

1. `configs/watermark_stage1.yaml`：理解 clean 三子阶段与最终视觉预算。
2. `configs/watermark_stage2.yaml`：理解 partial Encoder、七段退化课程和固定验证矩阵。
3. `train_watermark_diffusion.py`：查看双分支训练、AMP、验证与 checkpoint。
4. `models/watermark_unet.py`：查看 bits 的全局与空间双条件。
5. `models/watermark_residual.py`：查看内容 mask 和残差预算投影。
6. `models/watermark_decoder.py`：查看 residual multi-scale Decoder。
7. `NOISE_LAYER/`：查看四种可微退化。
8. `sample_embed_watermark.py` 与两个 eval 脚本：查看部署和评估口径。
9. `tools/debug_stage2_artifacts.py`：排查条纹、颜色伪影、频谱峰和梯度竞争。

## 17. 摘要

当前系统可概括为：

```text
Cover + 30-bit Message
  → cover-conditioned and watermark-conditioned diffusion redraw
  → content-aware bounded watermarked image
  → clean or four-device differentiable degradation
  → residual multi-scale decoder
  → recovered message
```

最关键的工程边界是区分单步训练代理与多步 DDPM 嵌入、保持 Stage 1/Stage 2 配置与 checkpoint 一致、严格检查权重加载提示，并使用固定退化矩阵而非单次随机攻击判断模型优劣。
