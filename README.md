# Micro-InternVL

GitHub: https://github.com/wksports/Micro-InternVL

基于官方 [InternVL3.5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B) 改进的开放词汇微生物显微图像目标检测器。

## 与官方代码的关系

本项目在官方 InternVL 开源代码（`internvl/` 目录）基础上做最小化修改：

- `internvl/model/internvl_chat/micro_internvl_head.py`：新增 Micro-InternVL 高分辨率 patch 检测头。
- `internvl/model/internvl_chat/modeling_internvl_chat.py`：在官方 `InternVLChatModel` 中接入检测头，新增 `forward_detection`、`encode_text_queries` 等方法；`extract_feature` 可选返回未压缩 patch 特征。
- `internvl/model/internvl_chat/configuration_internvl_chat.py`：新增 `micro_internvl_config` 配置字段。
- `micro_internvl/`：训练、评估、数据加载、损失函数等上层代码。

## 主要结构创新

1. **高分辨率 Patch 检测头**：在 InternViT 输出后、pixel-shuffle 压缩前（1024 patch token）接入检测头。
2. **跨尺度对比对齐**：patch-text / box-text InfoNCE 损失。
3. **层次化查询**：coarse / medium / fine 描述，训练时采样、推理时默认使用最细粒度。

## 目录结构

```
MicroDetect/
├── internvl/                       # 官方 InternVL 代码（已做 Micro-InternVL 改造）
│   └── model/internvl_chat/
│       ├── modeling_internvl_chat.py
│       ├── micro_internvl_head.py
│       └── ...
├── micro_internvl/                 # Micro-InternVL 训练/评估/数据代码
│   ├── config.yaml                 # 默认配置（兼容小显存）
│   ├── config_h20.yaml             # H20 优化配置
│   ├── model_wrapper.py
│   ├── train.py
│   ├── evaluate.py
│   ├── dataset.py
│   ├── losses.py
│   ├── queries.py
│   └── utils.py
├── scripts/                        # 一键脚本
│   ├── install.sh
│   ├── generate_queries.py
│   ├── train.sh
│   ├── eval.sh
│   ├── package.sh
│   └── push.sh                     # 上传到 GitHub
├── tests/
│   └── test_smoke.py
├── data/emds7/                     # EMDS-7 COCO 格式标注
├── checkpoints/                    # 保存的检查点
└── outputs/                        # 日志和评估结果
```

## 快速开始

### 1. 安装依赖

```bash
bash scripts/install.sh
```

### 2. 数据准备

将原始图像放在 `raw/emds7/EMDS7/`（与 `micro_internvl/config.yaml` 中的 `data.image_dir` 对应）。

也可以在服务器上直接下载公开 EMDS-7 图像数据：

```bash
python scripts/download_emds7.py --output-dir raw/emds7 --image-dir raw/emds7/EMDS7
```

脚本会显示下载、解压和图片整理进度，并检查 `data/emds7/instances_*.json` 中引用的图片是否都存在。

### 3. 生成层次化查询

```bash
python scripts/generate_queries.py --config micro_internvl/config.yaml
```

### 4. 训练

默认配置（适合 RTX 3080 等 10GB 级显存）：

```bash
bash scripts/train.sh
```

H20 服务器（96GB VRAM）推荐使用 H20 专用配置：

```bash
bash scripts/train_h20.sh
```

训练默认启用早停：验证集 `AP` 连续 30 个 epoch 不提升则自动停止，并保留最优检查点到 `checkpoints/final`。

### 5. 评估

```bash
bash scripts/eval.sh checkpoints/final test
```

H20 评估同样可指定 config_h20.yaml：

```bash
bash scripts/eval.sh checkpoints/final test --config micro_internvl/config_h20.yaml
```

## 配置说明

修改 `micro_internvl/config.yaml` 或 `micro_internvl/config_h20.yaml`：

- `model.base_model`: `OpenGVLab/InternVL3_5-4B`（GitHub 格式）
- `lora.r` / `lora.alpha`: 官方 backbone LoRA 参数
- `micro_internvl.detection_head`: 检测头结构
- `loss.lambda_patch` / `loss.lambda_box`: 对比对齐损失权重
- `data.image_dir`: 原始图像目录

H20 关键差异（见 `config_h20.yaml`）：
- `batch_size: 4`, `gradient_accumulation_steps: 2`
- `gradient_checkpointing: false`
- `num_workers: 8`

## 打包部署

```bash
bash scripts/package.sh
```

生成 `micro-internvl-deploy-YYYYMMDD.tar.gz`。

## 上传到 GitHub

```bash
bash scripts/push.sh
```

这会推送到 https://github.com/wksports/Micro-InternVL.git。确保你有该仓库的写权限。

## 本地测试

```bash
python tests/test_smoke.py
```

## 引用

- 官方 InternVL: https://github.com/OpenGVLab/InternVL
- InternVL3.5-4B: https://huggingface.co/OpenGVLab/InternVL3_5-4B
