# 真实多模态训练与 CLIP LoRA 实验

更新：2026-10-11。结果以 reports/clip_lora_summary.json、visual_summary.json 为准。没有追加整包图片下载。

## 两个独立任务与数据协议

现有官方图片对应590件商品，图片取得不是随机抽样，存在选择偏差。图文匹配按 product_code、图片SHA256和完全相同的文本联合分组，226组，413/93/84件训练/验证/测试商品，相关组不跨集合。并不保证排除所有视觉近重复。文本由真实名称、类型、颜色、描述构成，不生成标签。

购买召回沿用原时间切分，仅保留同时具有图片可用历史和目标的查询：1794训练、150验证、124测试。评测全590候选，历史截断10件，允许复购。它是条件子集评测，不能与41,585候选的主实验横比；过滤掉的购买目标也使该任务与原始任务不同。购买时间协议与图文商品分组协议是不同实验，不宣称购买召回的商品冷启动泛化。测试已探索性查看，不是盲测。

## CLIP 内部真实参数微调

实际复用 MIT 的 OpenAI CLIP ViT-B/32。recsys/clip_lora.py 在视觉和文本Transformer最后一层的两个MLP线性层注入LoRA：rank=4、alpha=8，共51,200个可训练参数。基础参数冻结，两个塔的适配参数均有实际非零梯度和变化；训练前后基础参数SHA保持一致。不是只训练外部投影，也不是生成式大VLM指令微调。

输入图片经官方预处理成[B,3,224,224]，文本token为[B,77]；两塔输出[B,512]归一化向量，得到[B,B]相似度。相同文本或图片按多正例处理，双向对比损失反传至LoRA，AdamW更新。3轮、batch16、lr1e-4，种子42/43/44，在MPS真实训练；按验证双向R@1均值选权重。

84候选测试：冻结图到文R@1=0.607143，LoRA三种子均值=0.690476；文到图由0.607143至0.615079，种子std=0.014848。这是小样本匹配结果，三种子不等于统计显著或业务收益。单次训练约19–23秒。结束MPS allocated约614MB、driver约1.64GB，均不是峰值显存。

保存best/last适配权重、逐轮日志、编码缓存和校验摘要。重新加载seed42权重，分别重编码训练/验证/测试一件，1e-5容差与缓存一致，见reports/clip_lora_checkpoint_check.json；这只是抽样一致性核查。

## 图文购买召回与消融

recsys/train_visual_retrieval.py 用真实购买历史训练用户塔、ID嵌入与图文投影：512维投影到64维，比较文本、图片、平均融合、门控融合。各模式10轮、batch64、同样负采样和验证选择；训练采样候选，验证测试全590候选，Recall/NDCG/MAP@12。

冻结和LoRA特征各4模式×3个召回种子，加3个ID基线共27次训练。LoRA迁移固定适配器seed42，三个召回种子不是三个适配器交叉实验。文本/图片模式也包含ID，不能解释成纯模态；ID基线历史架构不同，不是严格同容量消融。

测试NDCG@12三种子均值：ID 0.275858；冻结text/image/mean/gate分别0.291401/0.290838/0.291654/0.292176；LoRA分别0.285951/0.293219/0.289512/0.290878。复购基线0.307877。没有稳定融合收益或LoRA推荐迁移收益，保留负结果。按验证选的图文服务模型为冻结gate seed43，测试0.291888；部署模型不使用LoRA特征，LoRA是已完成的独立训练和迁移实验。

## 复现（已有合法数据和权重）

在项目根目录执行，依赖 requirements-hm.lock.txt。准备脚本会重建子集；先保留原始数据和reports。

```bash
.venv/bin/python -m scripts.prepare_visual_subset
for seed in 42 43 44; do
  .venv/bin/python -m recsys.train_clip_lora --device mps --seed "$seed"
done
for source in frozen lora; do
  for mode in text image mean gate; do
    for seed in 42 43 44; do
      .venv/bin/python -m recsys.train_visual_retrieval --source "$source" --mode "$mode" --seed "$seed"
    done
  done
done
for seed in 42 43 44; do
  .venv/bin/python -m recsys.train_visual_retrieval --source frozen --mode id --seed "$seed"
done
.venv/bin/python -m scripts.summarize_clip_lora
.venv/bin/python -m scripts.summarize_visual_experiments
.venv/bin/python -m scripts.verify_clip_lora_checkpoint
.venv/bin/python -m scripts.verify_visual_service
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m recsys.hm_serve --port 8766
# 服务运行后在另一终端
.venv/bin/python -m scripts.benchmark_hm_http --requests 100
```

## 本机交付与剩余限制

服务仅127.0.0.1:8766，visual为590候选图文子实验；semantic仍为41,585候选文本实验。默认复购。124条子集测试离线/服务Top12一致，版本fd16920e3efd；模型索引重编码一致性、权重SHA和错误处理纳入加载检查。

同版本本机HTTP短时warm测试，各策略并发1/4各100请求、5次预热，无错误。visual并发1 p50/p95=0.451/0.494ms，吞吐2160请求/秒；并发4 p95=3.312ms、1492请求/秒。CPU、共享MacBook Air M5 24GB、10条真实子集历史循环请求，不能外推为生产容量或SLA。完整结果reports/hm_http_fd16920e3efd.json，未测该进程峰值RSS。

下一步优先预声明新的时间窗口检验泛化、同架构融合对照、缺失模态鲁棒性与用户自己提出的假设；不为了简历扩大下载。当前没有大规模全目录图文训练、CTR/CVR、线上A/B或企业生产经历；590商品上的SmolVLM-500M生成式微调已完成，见[生成式实验](generative_vlm.md)，与本页CLIP任务分别评测。
