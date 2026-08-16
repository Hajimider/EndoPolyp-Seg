# EndoPolyp-Seg

## 项目说明

EndoPolyp-Seg 基于公开 Kvasir-SEG 内镜图像，完成息肉区域的像素级二值分割。项目使用轻量 U-Net 作为主模型，并增加 ResNet18 编码器迁移学习实验和 U-Net 输出后处理对照；OpenCV 阈值分割作为传统方法基线。完整流程包括图像与掩膜审计、固定数据划分、smoke 训练、验证集选模、测试集评估、ONNX 导出、CPU 推理测速和 Gradio 单图演示。

主模型在 150 张独立测试图像上取得 `Dice = 0.6562`、`IoU = 0.5366`、`Precision = 0.7951`、`Recall = 0.6562`。导出的 ONNX 模型大小为 `1.94 MB`，50 张测试图像上的 CPU 推理 p50 为 `23.05 ms`，p95 为 `29.94 ms`。这些数值来自项目当前保存的一次完整实验。

项目关注的是一套可复现的小型图像分割流程。模型使用公开数据训练，未经临床验证，输出不能作为医疗诊断依据。

## 实际问题与解决思路

内镜图像中的息肉形状、大小和颜色变化明显，边界还会受到反光、黏膜纹理和拍摄角度影响。项目将任务定义为：输入一张内镜图像，输出同尺寸的二值掩膜、边界叠加图、预测面积占比和单图推理耗时。

| 问题 | 项目处理方式 | 可检查的输出 |
| --- | --- | --- |
| 图像与掩膜可能缺失、损坏或尺寸不一致 | 双向核验文件名，检查可读性、尺寸和空掩膜 | `data/processed/report.json` |
| 需要得到像素级息肉区域 | 从零实现初始通道为 16 的轻量 U-Net | `artifacts/best.pt`、预测掩膜 |
| 需要说明深度模型是否真的有价值 | 使用 HSV/Otsu、形态学和最大连通域建立 OpenCV 对照 | `reports/classical_baseline.json` |
| 需要复用预训练视觉特征 | 增加 ResNet18 编码器 U-Net，冻结早期层并微调最后编码阶段 | `train.py --model resnet18_unet` |
| 模型输出存在小噪声和孔洞 | 在验证集选择阈值、最小连通域和闭运算核 | `reports/postprocess_tuning.json` |
| 训练和测试不能混用 | 验证集选择最佳权重，测试集只做最终报告 | 训练摘要与测试指标 |
| 运行设备没有 GPU | 使用 256×256 输入，在 CPU 上训练并导出 ONNX | 模型大小、p50/p95 延迟 |

## 数据与问题

Kvasir-SEG 包含 1,000 张胃肠道内镜图像及其像素级息肉掩膜。训练前会检查以下内容：

- `images` 与 `masks` 中的文件是否同名配对。
- 图像和掩膜是否可以正常读取，宽高是否一致。
- 掩膜二值化后是否包含有效前景。
- 每张图像只进入一个数据分区，避免同一文件跨分区使用。
- 记录息肉前景面积比例，了解大目标和小目标的分布范围。

本次数据审计找到 1,000 张图像和 1,000 张掩膜，全部有效，没有缺失配对、尺寸不一致或空掩膜。前景面积占比最小为 `0.57%`，中位数为 `11.76%`，最大为 `81.63%`。

## 数据处理与划分

项目固定随机种子 `42`，按图像进行 70% / 15% / 15% 划分：

| 数据集 | 图像数量 | 用途 |
| --- | ---: | --- |
| train | 700 | 训练 U-Net |
| val | 150 | 选择最佳权重和执行早停 |
| test | 150 | 最终一次独立评估 |

数据清单写入 `data/processed/manifest.csv`。清单保存图像、掩膜、原始尺寸、前景像素数和所属分区，后续训练、传统基线和评估脚本读取同一份划分。

## 数据链路

```text
Kvasir-SEG 原始图像与掩膜
  -> 文件配对、可读性、尺寸和有效掩膜检查
  -> 固定 seed=42 生成 train / val / test
  -> 写入 manifest.csv 和数据审计报告
  -> 1 epoch smoke 链路检查
  -> 轻量 U-Net 正式训练，验证集选择 best.pt
  -> ResNet18 编码器迁移学习 smoke / 独立实验
  -> 测试集只评估一次
  -> OpenCV 传统方法在同一测试集对照
  -> 验证集选择后处理参数，测试集比较前后指标
  -> 导出 ONNX 并检查数值一致性
  -> CPU 延迟测试与 Gradio 单图演示
```

## 模型与训练

### 轻量 U-Net

主模型从零实现，不使用预训练权重。编码器进行三次下采样，解码器通过转置卷积恢复分辨率，并使用 U-Net 跳跃连接融合浅层边缘信息和深层语义特征。初始通道数为 16，最终通过 `1×1` 卷积输出单通道 logits。

训练设置如下：

| 参数 | 默认值 |
| --- | ---: |
| 输入尺寸 | 256×256 |
| batch size | 8 |
| 最大 epoch | 25 |
| 初始学习率 | 0.001 |
| 优化器 | Adam |
| 损失函数 | BCEWithLogits + Dice Loss |
| 早停 patience | 6 |
| 随机种子 | 42 |

训练增强只使用随机水平翻转和垂直翻转。验证阶段不使用增强，以 `Dice` 选择并保存 `best.pt`。正式训练完成 25 个 epoch，最佳验证集 Dice 为 `0.6653`。本次训练曾从第 18 轮检查点继续，`training_summary.json` 记录完整轮数，当前历史 CSV 保留恢复后的第 19 至 25 轮。

### OpenCV 传统对照

传统方法读取 HSV 饱和度通道，使用 Otsu 自动阈值生成前景，再执行 7×7 椭圆核的开闭运算，最后保留最大连通域。它不参与 U-Net 训练，只用于观察固定阈值和颜色特征在复杂内镜图像上的局限。

### ResNet18 编码器迁移学习

在不改变二值分割任务的前提下，项目提供 `ResNet18 + U-Net 解码器` 作为改进实验。编码器使用 ImageNet 预训练权重，默认冻结 stem、layer1、layer2、layer3，只训练解码器和 layer4；也可以用 `--unfreeze-encoder` 解冻整个编码器。该实验使用独立的 `--tag` 产物前缀，不覆盖从零训练的 `best.pt`。

CPU 环境下先运行 smoke 验证结构和权重加载，再决定是否进行更长的独立训练。当前保存的 ResNet18 smoke 结果只用于链路检查，不与主模型测试指标混合。

### U-Net 输出后处理

后处理作用于模型输出的二值掩膜，不改变模型参数。`tune_postprocess.py` 在验证集搜索概率阈值、最小连通域面积和椭圆闭运算核；`postprocess_eval.py` 使用验证集选出的参数，在测试集并列报告原始掩膜和后处理掩膜。

当前验证集选择的参数为阈值 `0.35`、最小连通域 `128`、闭运算核 `5`。测试集原始 Dice 为 `0.6674`，后处理后为 `0.6681`；原始 IoU 为 `0.5467`，后处理后为 `0.5476`。参数只由验证集确定，测试集不参与搜索。

## 运行结果

以下主结果使用验证集选定的 `best.pt`，并按默认阈值 `0.5` 在测试集评估。后处理实验使用验证集单独选择的参数，结果在后文独立列出，避免混淆两种评估配置：

| 方法 | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| 轻量 U-Net | **0.6562** | **0.5366** | **0.7951** | 0.6562 |
| OpenCV 传统对照 | 0.2921 | 0.1892 | 0.1948 | **0.9192** |

OpenCV 对照的 Recall 很高，但 Precision 只有 `0.1948`，说明它把大量正常组织也分成了息肉。U-Net 的 Precision 和重叠指标明显更高，输出区域更集中；Recall 仍有提升空间，说明小息肉、低对比度区域和不规则边界仍可能漏分。

使用验证集选择阈值 `0.35`、最小连通域面积 `128` 和闭运算核大小 `5` 后，测试集 Dice 从同配置下的 `0.6674` 提升至 `0.6681`，IoU 从 `0.5467` 提升至 `0.5476`。提升幅度较小，说明后处理主要用于清理局部噪声，不能替代模型本身的分割能力。

### ONNX 与 CPU 推理

导出使用 ONNX opset 17，并在固定输入上比较 PyTorch 与 ONNX Runtime 输出：

| 项目 | 结果 |
| --- | ---: |
| ONNX 模型大小 | 1.94 MB |
| 最大绝对差 | `5.72e-06` |
| 平均绝对差 | `8.32e-07` |
| 输入尺寸 | 256×256 |
| 预热次数 | 10 |
| 计时图像 | 50 |
| p50 | 23.05 ms |
| p95 | 29.94 ms |
| 平均耗时 | 23.94 ms |

数值差异低于项目设置的 `1e-4` 上限。延迟只统计 ONNX 模型推理，实际页面耗时还会受到图像读取、缩放、后处理和当前硬件影响。

## 复现

### 1. 下载数据集

从 [Kvasir-SEG 官方页面](https://datasets.simula.no/kvasir-seg/) 下载，或直接下载 [Kvasir-SEG.zip](https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip)。解压后放到：

```text
EndoPolyp-Seg/
└── data/raw/Kvasir-SEG/
    ├── images/
    └── masks/
```

`images` 和 `masks` 中的文件需要同名配对。使用数据集前请阅读官方页面的许可和使用限制。

### 2. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 3. 一键运行

在项目根目录运行：

```powershell
python -u run_project.py
```

入口会依次检查数据清单、最佳权重、OpenCV 对照、测试报告、ONNX 文件和基准报告。缺少的步骤会自动执行，已有的对应产物会复用，最后启动 Gradio 页面。页面地址为 `http://127.0.0.1:7895`；IDE 没有自动打开浏览器时，可以直接访问该地址。

首次从零运行会包含完整 CPU 训练，需要等待较长时间。再次运行且产物齐全时会跳过训练。运行进度同时显示在控制台并写入 `run_project.log`。

### 4. 审计并划分数据

```powershell
python prepare_data.py
```

输出 `data/processed/manifest.csv` 和 `data/processed/report.json`。

### 5. 训练模型

```powershell
# 16 张训练图和 8 张验证图，运行 1 个 epoch，只检查训练链路
python train.py --smoke

# 正式训练
python train.py

# ResNet18 编码器迁移学习 smoke，不覆盖原 U-Net 产物
python train.py --model resnet18_unet --tag resnet18_smoke --smoke

# 独立进行迁移学习训练，可按 CPU 时间调整轮数
python train.py --model resnet18_unet --tag resnet18_tune --epochs 5
```

smoke 结果不能作为正式模型性能。训练意外中断时，可以从最近保存的检查点继续：

```powershell
python train.py --resume artifacts/last.pt
```

### 6. 运行传统对照和最终评估

```powershell
# 同一测试集上的 OpenCV 对照
python classical_baseline.py

# 使用最佳权重评估测试集并保存 6 组可视化样例
python evaluate.py
```

`evaluate.py` 默认不覆盖已有最终报告。确实需要重新评估时使用：

```powershell
python evaluate.py --overwrite

# 在验证集搜索后处理参数
python tune_postprocess.py

# 在测试集比较后处理前后指标
python postprocess_eval.py
```

### 7. 导出 ONNX 并测速

```powershell
python export_onnx.py
python benchmark.py
```

导出脚本会检查 PyTorch 与 ONNX 输出差异；基准脚本默认预热 10 次，并在 50 张测试图像上计时。

### 8. 启动演示

```powershell
python app.py
```

页面上传单张内镜图像后，返回预测掩膜、边界叠加图、预测面积占比和 ONNX 推理耗时。直接运行 `app.py` 前需要已有 `artifacts/best.onnx`；没有模型时先运行导出脚本或一键入口。

## IDE 运行

PyCharm 或 VS Code 打开 `EndoPolyp-Seg` 目录后，由使用者选择已经安装项目依赖的 Python 环境。项目不绑定本机解释器路径、Conda 环境名或 SDK 名称。

| 入口 | 用途 |
| --- | --- |
| `run_project.py` | 检查并补齐流程，最后启动 Demo |
| `prepare_data.py` | 数据配对审计和固定划分 |
| `train.py` | smoke 或正式训练 U-Net |
| `train.py --model resnet18_unet` | ResNet18 编码器迁移学习实验 |
| `classical_baseline.py` | OpenCV 传统分割对照 |
| `evaluate.py` | 验证或测试指标与预测样例 |
| `tune_postprocess.py` | 验证集选择后处理参数 |
| `postprocess_eval.py` | 测试集比较后处理前后指标 |
| `export_onnx.py` | ONNX 导出和数值一致性检查 |
| `benchmark.py` | ONNX Runtime CPU 延迟测试 |
| `app.py` | Gradio 单图演示 |

项目提供 `.run/` 和 `.vscode/launch.json` 运行配置，所有脚本都使用项目相对路径。PyCharm 可直接选择 `EndoPolyp: Run All`，VS Code 可在“运行和调试”中选择同名配置。

## 项目结构

```text
EndoPolyp-Seg/
├── run_project.py                 # IDE/命令行一键入口
├── app.py                         # Gradio 单图分割演示
├── prepare_data.py                # 图像/掩膜审计和固定划分
├── train.py                       # U-Net smoke / 正式训练
├── tune_postprocess.py            # 验证集后处理参数搜索
├── postprocess_eval.py            # 后处理前后测试集对比
├── classical_baseline.py          # OpenCV 传统分割对照
├── evaluate.py                    # Dice/IoU 等指标和预测样例
├── export_onnx.py                 # ONNX 导出和一致性检查
├── benchmark.py                   # ONNX Runtime CPU 测速
├── src/polypseg/                  # 数据、模型、指标和推理模块
├── tests/                         # 单元测试
├── data/
│   ├── raw/Kvasir-SEG/            # 本地下载的数据集
│   └── processed/                 # 数据清单和审计报告
├── artifacts/                     # 本地权重和 ONNX 文件
├── reports/                       # 指标、历史和预测样例
├── .run/                          # PyCharm 共享运行配置
├── .vscode/launch.json            # VS Code 共享运行配置
├── requirements.txt
└── README.md
```

目录树用于说明各文件和目录的职责。

## 输出文件

```text
data/processed/report.json                  # 数据审计与分区数量
data/processed/manifest.csv                 # 图像/掩膜路径和固定分区
artifacts/smoke.pt                          # 1 epoch 链路检查权重
artifacts/best.pt                           # 验证集 Dice 最佳权重
artifacts/last.pt                           # 最近一轮训练检查点
artifacts/best.onnx                         # Gradio 使用的 ONNX 模型
reports/smoke_summary.json                  # smoke 训练摘要
reports/training_summary.json               # 正式训练摘要
reports/training_history.csv                # 训练历史
reports/classical_baseline.json             # OpenCV 测试集指标
reports/evaluation.json                     # U-Net 测试集指标
reports/prediction_samples/                 # 原图、真值、预测和叠加图
reports/onnx_validation.json                # PyTorch/ONNX 数值差异
reports/onnx_benchmark.json                 # ONNX Runtime CPU 延迟
reports/postprocess_tuning.json             # 验证集后处理参数
reports/postprocess_evaluation.json         # 后处理前后测试集指标
```

## 测试

```powershell
python -m pytest -q
```

当前 13 个测试覆盖数据配对与固定划分、无效样本审计、图像张量归一化、掩膜二值化、Dice/IoU/Precision/Recall、U-Net 输出尺寸、ResNet18 U-Net 输出尺寸、Dice Loss、后处理连通域与闭运算和 ONNX/PyTorch 输出一致性。

## 局限

- Kvasir-SEG 只有 1,000 组图像和掩膜，数据来源、成像设备和息肉类型有限，测试结果不能直接外推到其他医院或设备。
- 当前按图像随机划分。数据集没有在本项目中提供可用的患者或视频分组信息，因此不能证明不同分区来自完全独立的患者。
- 主模型从零训练且规模较小；ResNet18 迁移学习只作为独立改进实验，没有与主模型结果混写，也没有使用多模型融合。
- OpenCV 方法依赖颜色和最大连通域，只适合作为可解释对照，不是稳定的息肉分割方案。
- 默认模型评估使用阈值 0.5；后处理实验在验证集选择阈值 0.35、最小连通域面积和闭运算参数，再在测试集独立比较。参数仍可能随数据来源和应用场景变化，需要重新校准。
- CPU 延迟受处理器、线程数和运行时版本影响，本文数字只代表本次基准运行。
- 项目没有覆盖数据采集、临床标注复核、跨中心验证或医疗器械合规，不用于临床诊断。

## English Summary

EndoPolyp-Seg is a reproducible endoscopic polyp segmentation project based on the public Kvasir-SEG dataset. It audits 1,000 image-mask pairs, creates a deterministic 70/15/15 split, trains a small U-Net from scratch, compares it with an OpenCV threshold-and-morphology baseline, evaluates the test split once, exports ONNX, benchmarks CPU inference, and provides a Gradio single-image demo. The U-Net achieved a test Dice of 0.6562 and IoU of 0.5366. ONNX Runtime p95 latency was 29.94 ms on the recorded 50-image benchmark. The model is intended for algorithm practice and portfolio demonstration, not clinical diagnosis.

## 参考资料

- [Kvasir-SEG 数据集](https://datasets.simula.no/kvasir-seg/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Gradio](https://www.gradio.app/)
