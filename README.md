# 多模态商品推荐与 VLM 实验

基于 H&M 真实购买数据研究商品召回与个性化排序，包含 CLIP 图文对齐/LoRA 和 SmolVLM 属性生成扩展，另保留可在普通电脑运行的 VOGUE 轻量演示。

**首次运行请看：[从零部署与代码导读](docs/getting_started.md)**。

## 内容

- H&M：10 万用户、1,342,361 条窗口交易、41,585 件训练候选商品；对比复购、热门、ID 双塔和 CLIP 文本召回及排序。
- 图文子实验：590 件商品，CLIP 内部 LoRA、多种特征融合与购买召回消融。
- 生成式扩展：SmolVLM-500M 类别/颜色 JSON 生成、LoRA 与视觉 token 保留实验。
- VOGUE：轻量特征、双塔召回、排序、离线评测与本机 HTTP 服务，适合先跑通完整工程链路。

历史比较结果支持复购基线作为 H&M 默认推荐策略。大目录、图文子集与生成式任务分别评测，不能直接混用指标。报告反映当时运行结果，不代表每台机器上的实时性能。

## 本地运行

需要 Python 3.12、Git 和网络。先克隆仓库，然后按部署指南下载数据、训练模型并启动服务。普通电脑建议先运行 VOGUE；H&M 与 VLM 需要额外数据、模型和计算资源。

```bash
git clone https://github.com/gaotianding18-hub/multimodal-recsys.git
cd multimodal-recsys
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.lock.txt
python scripts/download_vogue_data.py
python scripts/prepare_vogue_ratings.py
python scripts/audit_vogue.py
python -m recsys.train
python -m recsys.serve --port 8765
```

打开 http://127.0.0.1:8765 。完整环境说明、H&M 启动方法、API 验证及排错见 [部署指南](docs/getting_started.md)。

## 目录与阅读顺序

1. `configs/`：数据协议和训练配置。
2. `scripts/`：下载、数据准备、实验编排和验证工具。
3. `recsys/`：特征、召回、排序、LoRA、评测和 HTTP 服务。
4. `web/`：推荐演示与商品属性页面。
5. `reports/`：历史实验指标及核查记录。
6. `tests/`：协议、模型和服务一致性检查。

深入阅读：[架构](docs/architecture.md)、[H&M 代码导读](docs/hm_code_guide.md)、[实验总览](docs/experiment_overview.md)、[图文实验](docs/multimodal_experiments.md)、[生成式实验](docs/generative_vlm.md)。

## 数据、产物与使用条件

仓库公开源码、配置和技术实验记录。原始数据、独立商品图片、训练权重、虚拟环境、凭据及个人职业/学习资料未包含在此发布快照中，旧文档涉及这些本机资料的链接仅是历史记录。克隆后不能直接复用作者机器的模型，需要按说明获取数据并训练。训练会更新部分报告，建议先备份历史报告。

服务绑定本机回环地址。原验证环境为 Apple Silicon macOS，其他设备的性能与兼容性需自行验证。VOGUE 轻量实现使用像素/词哈希特征；CLIP 与 SmolVLM 是独立扩展。没有线上 A/B 或 CTR/CVR 结果。

第三方代码、数据、模型的署名和使用条件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。代码公开不改变数据和模型的许可限制。
