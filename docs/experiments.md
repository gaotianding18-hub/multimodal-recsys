> 2026-10-12：本文包含历史阶段记录。当前工程与实验结论以[实验总览](experiment_overview.md)为入口；原始reports保留，不因教学更新而覆盖。

# 真实实验与失败分析

任务与分母见docs/protocol.md。测试12次情境，仅2名用户；以下结果不能证明统计显著或业务效果。

## 基线

- train_mean_rating: NDCG@3=0.4270，Recall@3=0.4000。
- scenario_text_similarity: NDCG@3=0.1572，Recall@3=0.0000。

## 3个种子消融

- text/retrieval_test: NDCG@3=0.3530 ± 0.0494，Recall@3均值=0.3667。±为3种子样本标准差，不是置信区间。
- text/reranked_test: NDCG@3=0.3603 ± 0.0468，Recall@3均值=0.3333。±为3种子样本标准差，不是置信区间。
- image/retrieval_test: NDCG@3=0.4015 ± 0.0895，Recall@3均值=0.4333。±为3种子样本标准差，不是置信区间。
- image/reranked_test: NDCG@3=0.3832 ± 0.0373，Recall@3均值=0.4333。±为3种子样本标准差，不是置信区间。
- multimodal/retrieval_test: NDCG@3=0.3929 ± 0.0187，Recall@3均值=0.4333。±为3种子样本标准差，不是置信区间。
- multimodal/reranked_test: NDCG@3=0.3940 ± 0.0160，Recall@3均值=0.4333。±为3种子样本标准差，不是置信区间。

服务按验证选择text-43，测试端到端NDCG@3=0.4044。默认商品塔为文本消融版本，用户输入仍包含入门偏好图像。可用--run multimodal-42体验双模态商品塔。

## 失败与限制

- 平均评分基线优于默认复杂模型；融合没有稳定优势，排序部分运行退化。不得写提升。
- 输入像素低分辨率、词哈希语义弱；只有6名训练用户，容易拟合个体与场景，验证选择的多次比较也有过拟合风险。
- 随机种子变化不替代更多独立用户；不从测试调参。
- 无正例的2个测试情境仍计入分级NDCG，但不计入Recall分母。
- 图片完全重复及dHash近重复候选审计未发现重复；启发式检查不能证明完全无近重复。
- VLM2Vec主干PEFT、TorchEasyRec官方安装、多规模数据与用户独立改进尚未完成，本版仅借鉴思想。

## 资源与复现

固定9组CPU实验宿主复跑23.19秒，峰值RSS353894400字节，内存footprint291537952字节（/usr/bin/time -l）。首次沙箱测量因sysctl权限报错，失败记录保留，不虚构GPU显存。

训练配置、完整每轮日志及权重见configs/training.json与artifacts/runs；实验汇总见reports/experiments.json。
