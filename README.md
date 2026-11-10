# rPPG 3D CNN

配置驱动的脉率估计工程。合成数据和有参考脉率的视频共用同一个 ConvNet3D。`weights/best.pt` 来自原项目的 `local_rtx2060_refresh_seed42`，可以直接用于测试、推理或微调初始化。

## Ubuntu CPU 环境

将工程目录复制到 Surface，进入该目录运行。以下用 Ubuntu 24.04 和 Python 3.12。Python 3.11 及以上用内置 `tomllib`，Python 3.10 通过依赖清单安装 `tomli`。

```bash
sudo apt update
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## 先用已有权重测试

在 Surface 上运行合成数据命令前，将 `configs/synthetic.toml` 的 `[runtime]` 中的 `device` 设为 `"cpu"`，可先把 `[train]` 中的 `batch_size` 设为 `1`。

```bash
python -m rppg evaluate --config configs/synthetic.toml --checkpoint weights/best.pt
```

默认测试 75 个合成样本，终端会打印 `run_dir`。将 `[data.test]` 中的 `samples_per_class` 改为 `10` 可测试 750 个样本。训练、验证和测试各自设置 `source`。`synthetic` 根据 `seed` 和 `samples_per_class` 生成样本，`real` 从指定的清单取对应划分的视频窗口。`[train].batch_size` 也用于测评，Surface 起步用 `1`。

## 训练、自动测试和 loss 图

从工程目录运行训练命令。合成数据配置默认从头训练，真实微调配置使用 `init_checkpoint` 指定的权重。

```bash
python -m rppg train --config configs/synthetic.toml
```

两个训练配置的 `[train]` 段都已开启自动测试。自己新建配置时，把下面的参数放在现有的 `[train]` 段中，并配置 `[data.test]`。

```toml
[train]
test_after_train = true
```

训练每轮记录训练和验证 loss，并按验证 loss 保存 `best.pt`。训练结束后用 `best.pt` 测试一次，`loss_curve.png` 显示训练曲线、验证曲线和最佳轮次上的最终测试 loss 点。将 `test_after_train` 设为 `false` 时，可以省略 `[data.test]`，训练结束后仍会绘制训练和验证曲线。

终端会打印 `run_dir` 和 `loss_plot`。用打印的 `run_dir` 替换示例路径，便可查看 loss 图、测试指标，或者为测试预测另生成脉率图。

```bash
TRAIN_RUN="runs/<experiment>/<本次训练目录>"
ls "$TRAIN_RUN/loss_curve.png"
cat "$TRAIN_RUN/test/metrics.json"
python -m rppg visualize --predictions "$TRAIN_RUN/test/predictions.csv"
```

后两条命令使用自动测试的输出。`visualize` 生成预测结果图，训练 loss 图由 `train` 自动保存。

## 用有参考脉率的视频微调和测评

把视频放入 `datasets/real/videos/` 并填写 `datasets/real/manifest.csv`。每行是一个窗口，下面是格式示例。

```csv
subject_id,session_id,video_path,start_frame,roi_x,roi_y,split,label_kind,reference_bpm
p01,s01,videos/p01.mp4,0,120,60,train,pulse,74.2
p02,s01,videos/p02.mp4,30,110,58,val,pulse,71.8
p03,s01,videos/p03.mp4,60,124,65,test,pulse,79.1
```

- 视频应为恒定约 30 fps。`start_frame` 从 0 开始，从这一帧起读 60 帧。
- `roi_x`、`roi_y` 是一个固定 25×25 皮肤图像块的左上角。微调和带标签测评初版每窗口使用一个块。
- `reference_bpm` 应来自同一窗口的同步参考数据。`pulse` 标签取最近的模型脉率档位，测评误差使用原始参考值。模型档位范围为 55–237.5 BPM。明确标注为无脉搏信号的窗口用 `label_kind=no_pulse`，BPM 留空。
- 真实划分共用一份清单。同一受试者或视频只能出现在一个划分中。同一段视频的多个窗口可以属于同一个划分。
- TOML 中的路径相对 TOML 文件。清单内的视频路径相对清单文件。命令行的 `--checkpoint` 相对当前目录。

```bash
# 测试原权重，作为真实数据基线
python -m rppg evaluate --config configs/real_finetune.toml --checkpoint weights/best.pt

# 加载原权重，冻结卷积层，训练分类头
python -m rppg train --config configs/real_finetune.toml

# 如需单独复测，将 TRAIN_RUN 替换为训练命令输出的 run_dir
python -m rppg evaluate --config configs/real_finetune.toml --checkpoint "$TRAIN_RUN/best.pt"
```

`init_checkpoint` 指定微调的初始权重。设为 `freeze_backbone = false` 可训练整个网络。`fps`、`window_frames`、`patch_size` 当前固定为 30、60、25，以匹配已有网络和权重。

## 直接读取 rPPG-10 视频和 ECG 测试

将 `Dataset_rPPG-10` 和工程目录放在同一级，或在 `configs/rppg10_test.toml` 中填写实际的 `video_path`、`ecg_path`。已安装旧版依赖时，先重新执行 `python -m pip install -r requirements.txt`，为 ECG 处理安装 SciPy。

```bash
python -m rppg evaluate --config configs/rppg10_test.toml --checkpoint weights/best.pt
```

测试某次训练得到的权重时，直接传入该实验目录下的 `best.pt`。

```bash
python -m rppg evaluate --config configs/rppg10_test.toml --checkpoint runs/synthetic/20261109-233741-train-hd4bygg0/best.pt
```

对于权重旁有 `training.csv` 的训练实验，单独执行的 rPPG-10 测评会在该训练目录的 `evaluate/<output.experiment>/<本次测评目录>/` 下生成结果。上面的命令会把结果归到 `20261109-233741-train-hd4bygg0/evaluate/rppg10_subject1_forehead/` 下。使用集中存放的 `weights/best.pt` 时，结果按配置中的 `[output]` 保存。终端的 `run_dir` 给出本次测评目录。

`[data.test]` 的 `source = "rppg10"` 使用整段 ECG 检测 R 峰，再按每个视频窗口计算参考 BPM。配置中的 `start_frame` 从 0 开始，`stop_frame` 是不含的右端，`stride_frames` 是两个窗口起点的间隔。`ecg_offset_s` 加到视频时间上得到 ECG 时间，默认 `0.0` 沿用发布数据的起点对齐。换受试者时同时修改 `subject_id`、视频和 ECG 路径。对比同一受试者的前额和脸颊时，换 AVI 并保持 ECG 和窗口设置一致。

参考 BPM 在测评时计算，出现在 `predictions.csv` 的 `reference_bpm` 列。`sample_id` 末尾的 `frame-数字` 是窗口的起始帧。`metrics.json` 的 `reference_windows` 记录候选窗口、有效标签和各类跳过数量。覆盖不足、RR 间隔不合格或参考 BPM 超出模型档位的窗口会跳过。测评过程使用内存标签，运行目录保存正常的测评结果。

## 视频和摄像头推理

在 `configs/video_predict.toml` 设置 `video_path` 和 `roi = [x, y, width, height]`。推理会把区域切成 25×25 的块并汇总脉率。区域越大，需要计算的块越多。

```bash
python -m rppg predict --config configs/video_predict.toml --checkpoint weights/best.pt
```

把 `source` 改为 `camera` 就用 `camera_index` 对应的摄像头。实时模式在终端输出 BPM，并逐窗口写入 CSV。默认用最近 60 帧，每新增 30 帧提交一个窗口。推理繁忙时跳过新的任务，避免累积延迟。到帧时间估计的采样节奏不符时写入 `invalid_timing`；无有效块时写入 `no_pulse`。按 Ctrl+C 停止。Surface 的摄像头可用性和持续运行速度需要在设备上验证。

## 输出和可视化

默认每次运行生成独立目录 `runs/<experiment>/<时间>-<操作>-<标识>/`。直接指定训练实验权重的 rPPG-10 测评按上节说明归档。`config.json` 记录本次实际配置与权重路径。

| 操作 | 输出 |
| --- | --- |
| `train` | `best.pt`、`training.csv`、`loss_curve.png`；启用自动测试时另有 `test/metrics.json` 和逐样本结果 |
| `evaluate` | `metrics.json`、`predictions.csv`；真实数据另有 `per_video_metrics.csv` |
| `predict` | `predictions.csv`，包含窗口时间、脉率、状态、有效块数 |
| `visualize` | `plots/` 下的脉率曲线、参考值散点图和误差直方图，视 CSV 中的字段值而定 |

```bash
# 将 EVAL_RUN 替换为测评或推理输出的 run_dir
python -m rppg visualize --predictions "$EVAL_RUN/predictions.csv"
```

多段视频默认绘制第一段的时间曲线。可用 `--video-id` 选择 CSV 中的一段视频。误差只在同时具有参考脉率和有效预测的样本上计算，结合 `pulse_coverage`（有参考脉率的样本中成功输出的比例）阅读。合成数据指标用于检查合成基线，真实数据精度由独立的带标签视频测评。
