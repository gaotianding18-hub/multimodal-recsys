# 从零部署与代码导读

本页面向首次克隆项目的朋友。所有命令在仓库根目录执行。

## 1. 项目做什么

项目有三条相互关联的路线：

- **H&M 购买推荐（主线）**：历史购买 → 候选召回 → 个性化排序 → 商品页面。对比复购、热门、ID 双塔及 CLIP 特征，保留复杂模型未胜过基线的结果。
- **CLIP 图文子实验**：590 件商品的图文对齐、内部 LoRA、购买召回消融。它与 41,585 件商品的大目录评测有不同候选范围。
- **SmolVLM 生成式扩展**：商品图片 + 指令 → 类别/颜色 JSON，包含 LoRA 和视觉 token 保留实验。

另外保留一个 **VOGUE 轻量演示**，适合先在普通电脑跑通数据、训练、评测、服务。它使用像素/词哈希特征，不是 CLIP 或 VLM 大模型。

仓库中的 `reports/` 是原机器的历史实验记录，不是下载好的模型。`data/`、`artifacts/`、虚拟环境和凭据均不上传。原机器的数据约 5.3 GB、模型及产物约 2.8 GB；完整官方数据包可能更大，不能以这两个数字估算整包下载空间。

## 2. 首次运行：推荐先跑 VOGUE

需要 Git、Python 3.12 和网络；原验证环境为 Apple Silicon macOS。CPU 可以运行轻量版；Windows 建议 WSL2 按以下 Linux 命令执行。其他平台未做完整验证。

```bash
git clone https://github.com/gaotianding18-hub/multimodal-recsys.git
cd multimodal-recsys
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.lock.txt
python scripts/download_vogue_data.py
python scripts/prepare_vogue_ratings.py
python scripts/audit_vogue.py
python -m recsys.train
python -m recsys.serve --port 8765
```

浏览器打开 http://127.0.0.1:8765 。用户可选 `s1` 至 `s10`，场景 `1` 至 `6`。Ctrl+C 停止服务。若需显式查看商品图文双模态塔，停止后执行：

```bash
python -m recsys.serve --port 8765 --run multimodal-42
```

另开终端验证：

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"s9","scenario":1,"k":3}'
```

下载器依据 `reports/vogue_manifest.json` 获取固定版本并核验 Git blob 哈希。网络失败需修复网络后重试，不要跳过数据检查。训练会生成 `artifacts/runs/`，并更新 `reports/experiments.json`；如需保留原报告，先复制整个 `reports/` 到仓库外。

固定依赖来自原机器；若包源缺少某个精确版本，应核查包源与 Python 版本，不建议悄悄更换版本后声称严格复现。只想做兼容性尝试可使用 `requirements-local.txt` 的三项核心依赖，但需自行记录实际版本并重新验证。

## 3. H&M 主线：重新取得数据并训练

先到 [H&M 官方 Kaggle 比赛](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)，使用自己的账号按官方规则取得数据。代码公开不代表数据和图片可随意再分发。将 `articles.csv`、`customers.csv`、`transactions_train.csv` 放到 `data/hm/raw/`。凭据配置见 [Kaggle 说明](kaggle_local_setup.md)，不要借用作者的密钥。

在上一步虚拟环境中安装主线依赖，然后执行：

```bash
python -m pip install -r requirements-hm.lock.txt
python scripts/audit_hm.py
python scripts/prepare_hm.py
python scripts/run_hm_experiments.py
python -m recsys.hm_rank
```

这一步包括六组 ID 召回实验和所选模型的排序训练，耗时和内存远大于轻量演示，取决于设备。训练脚本只有在报告与模型、索引都存在时才跳过已有运行。

**首次只运行 ID/复购服务时**，仓库自带 `reports/visual_summary.json` 会使服务尝试加载原机器的图文模型。若尚未复现图文实验，先把这个历史选择报告移入备份目录（不修改作者原始实验数据）：

```bash
mkdir -p artifacts/historical-reports
mv reports/visual_summary.json artifacts/historical-reports/
python -m recsys.hm_serve --port 8766
```

打开 http://127.0.0.1:8766/ ，检查 `http://127.0.0.1:8766/health` 返回的可用方法。没有下载商品图片时页面无法展示真实图片；ID 与复购推荐不依赖 CLIP 图片特征。模型、目录、配置不一致会被服务拒绝，应重新生成匹配产物。

进一步复现大目录 CLIP 文本、590 商品图文和生成式任务，分别遵循：

- [H&M 部署细节](hm_deployment.md)
- [图文数据、CLIP LoRA 与训练步骤](multimodal_experiments.md)
- [SmolVLM 属性生成与视觉 token 实验](generative_vlm.md)

其中 `--device mps` 仅用于支持 MPS 的 Mac；其他设备按对应脚本的 `--help` 选择 CPU/CUDA。图文服务需要重新生成的匹配模型、特征、目录与选择报告；不要直接恢复历史选择报告来代替训练。生成式 API 使用 8767 端口，需要单独下载基础模型并生成训练产物。

## 4. 如何读懂代码

建议从轻量链路理解接口，再看主线：

1. `configs/training.json`、`configs/hm.json`：切分、样本窗口和训练参数。
2. `scripts/prepare_vogue_ratings.py`、`scripts/prepare_hm.py`：原始数据如何变成用户历史、候选和标签。
3. `recsys/core.py`、`recsys/train.py`：轻量特征、双塔召回、排序、指标与模型保存。
4. `recsys/hm_train.py`、`hm_rank.py`、`hm_hybrid.py`：主线召回、负采样、排序和基线。
5. `recsys/clip_features.py`、`clip_lora.py`、`train_clip_lora.py`、`hm_multimodal.py`：CLIP 编码、LoRA 和多模态召回。
6. `recsys/train_generative_vlm.py`、`vlm_serve.py`：生成式训练与推理。
7. `recsys/serve.py`、`hm_serve.py` 与 `web/`：HTTP 接口和前端页面。

整体流程是：原始数据 → 按协议切分/特征构造 → 训练召回 → 训练排序 → 保存模型与索引 → 服务校验并加载 → 推荐/生成接口。分数是相对偏好，不是购买概率。

详细阅读入口：[架构](architecture.md)、[H&M 代码导读](hm_code_guide.md)、[实验总览](experiment_overview.md)。个人教学/面试材料未纳入公开快照，朋友部署以本页为入口。

## 5. 验证与常见问题

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

完整测试包含读取 H&M 模型的集成检查；只跑轻量版时，缺少这些产物的测试无法通过。不要据此声称完整实验已复现。

- 找不到 `data/` 或 `model.pt`：克隆只带源码，先完成相应下载、准备与训练。
- 端口占用：停止原服务或通过 `--port` 更换端口。
- 模块无法导入：回到仓库根目录，并使用同一个虚拟环境安装依赖和执行命令。
- 401/403/429：核查数据授权、凭据或等待限流恢复；不要将凭据提交 Git。
- 历史报告与新结果不同：记录实际依赖、设备、种子与数据版本；历史报告不能作为本次运行证明。

所有服务默认绑定本机回环地址，适合本地研究演示。第三方代码、数据和模型的使用条件见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
