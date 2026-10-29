# 可行性核查与实施方案

核查日期：2026-10-11（Asia/Shanghai）。“已核实”“推断”“待测”分别记录；不将计划当结果。

## 环境（已核实）

工作区初始为空，无适用的父目录 AGENTS.md。MacBook Air，Apple M5，10 核 CPU、10 核 GPU、24 GB 统一内存，arm64 macOS，约 329 GiB 可用空间。系统 Python 3.9.6；另有 Python 3.12.14 和 uv，项目采用后者隔离环境。统一内存由系统与 GPU 共享，不是 24 GB 独占显存。

用户要求自行估计本机性能；当前按仅本机、云预算未授权、私有本机演示推进。每周时间和截止日期未知，以验收节点排期。

实际探针：PyTorch 2.8.0 在沙箱内 MPS 不可用，宿主权限复核 MPS 可用。小网络 [128,1024]→256→128、5步预热、50步AdamW训练，MPS每步约0.472ms；100批推理平均约0.208ms/批。最终 MPS 张量分配约6.38MiB、驱动分配约34.72MiB，不是峰值。见 reports/environment-host.json。这是短时随机张量探针，无真实数据读取/推荐评测，不能外推图文编码、大模型训练时间或服务吞吐；后续 GPU 训练须使用能够访问宿主设备的执行权限。

推断：可承担小规模数据清洗、冻结图文编码器批量编码、投影层/用户塔训练、MLP 排序和小索引。MPS 可用性及算子实际性能见运行后的 reports/environment.json。MacBook Air 长时训练可能受到散热与共享内存影响，不从短探针推断长时吞吐。

先尝试 5千商品/5万交互以内、128维表示、batch 64、1 epoch；这是上限建议，必须在数据检查后确定。5千×128×4 字节的纯向量约 2.44 MiB，不包含 ID、索引结构、模型和缓存。3B 参数仅 FP16 权重约 6 GB，完整训练还需梯度、优化器和激活，不推荐直接在此机器训练。LoRA 也不能消除基础权重与激活开销。正式时间/峰值内存不预报，先测真实批次。

## 参考仓库

### TorchEasyRec

官方：https://github.com/alibaba/TorchEasyRec

已核实：master，未归档，最新默认分支提交 94d9d953994f2820799c49599b3827f915683ba5（2026-10-09）；仓库 pushed_at 是 2026-10-11，不等同于默认分支提交时间。Apache-2.0。文档入口与本地训练说明存在。

固定提交 requirements/runtime.txt 包含 torch==2.13.0、torchrec==1.8.0、fbgemm-gpu==1.8.0、transformers>=4.56，graphlearn/pyfg 的直接 wheel 指向 Linux x86_64。官方本地教程使用 CUDA 安装路径。不能把“CPU支持”视为“macOS arm64 支持”。尚未做完整安装和模型运行验证。

推荐：参考双塔、DIN/DeepFM 思路、配置与训练服务一致性设计；本地先自行实现一个 MLP 排序器，必要时再加 DIN，不复制生产基础设施。未来有 Linux GPU 再评估单独环境跑官方模型。源码路径需按当前文件树定位，旧 tzrec/models/din.py 已返回404。

### VLM2Vec

官方：https://github.com/TIGER-AI-Lab/VLM2Vec

已核实：main，未归档，提交 d00b5dcdeaa099f9ddb221ea081128d609c22acb（2026-09-21），Apache-2.0。当前 README 已围绕 MMEB-V3 展开；train.py 使用 GradCacheLateProcessTrainer。requirements 含 flash-attn、decord、transformers==4.52.3、numpy==1.26.4，大量包未固定版本。

推断：官方依赖栈不适合直接安装到当前 Mac；与 TorchEasyRec 主分支 Transformers 约束冲突。二者只通过稳定的商品 ID/特征文件对接，不共享官方全量环境。

推荐最小替代：小型图文编码器冻结编码 → 训练图文投影与融合层 → 历史聚合用户塔 → 对比学习。它属于任务相关多模态表示训练，不应称为完整 VLM2Vec 复现或大 VLM 微调。若用户坚持 VLM 主干 PEFT，应作为算力决策单独确认。基础模型权重许可证另核实，代码许可不覆盖模型与数据。

维护情况依据提交时间，不能据此保证稳定性。已定位并读取 MultiTowerDIN 使用 DINEncoder 的排序实现；VLM2Vec 的当前模型路径是 src/model/model.py 与 src/model/biencoder_gc.py，读取请求超时，对其损失源码核查仍未完成。原始核查与失败请求保存在 reports/upstream_snapshot.json、reports/source_inspection.json、reports/model_inspection.json；不声称已完整审阅源码或跑通官方框架。

## 数据可行性与尚未解决的门槛

候选 A：[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)，[官方数据卡](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)。评论与商品 metadata 可按 parent_asin 对齐；不能以变体 asin 直接联接。商品标题/features/description 和 metadata.images 可用；评论图片是购买后的用户内容，不作为预测该交互的输入。评论时间不是购买时间，verified_purchase 不提供曝光或购买日志。

官方全量约 571.54M 评论；All_Beauty 约 701.5K 评论、632K 用户、112.6K 商品，行为较稀疏，不应假设筛选后足够训练。下载为官方 review/meta JSONL 压缩文件，优先流式读原文件，不执行远程加载脚本。文档示例时间字段存在 timestamp/sort_timestamp 差异，须实测字段和单位。图片是 URL，实际可达率、重复率、授权范围未验证。

限制：所查官方页面没有足以确认的明确数据许可，不采用第三方镜像自声明许可证替代原始授权。因此 A 是技术优先候选，尚未批准为正式训练数据，也未下载全量数据。商品内容为抓取时快照，无法证明历史版本；即使用时间切分，仍只能报告“静态目录回溯评测”，不能宣称彻底消除所有时间泄漏。禁止 average_rating、rating_number、bought_together 等跨期汇总特征进入模型。

候选 B：[MerRec 官方数据卡](https://huggingface.co/datasets/mercari-us/merrec)：明确 CC BY-NC 4.0，适用于非商业学习研究并须署名。按月 Parquet 分片，总量约166 GB/1.27B事件，只考虑小分片。含 user_id/item_id/stime/event_id/name 与商品属性；已查字段没有商品图片，不能从其他商城强行拼图。item_view/like/buy事件并不自动构成完整曝光集，不能未经窗口与标签核实称 CTR/CVR。

用户已明确选择保留多模态目标，继续核实统一图文交互数据，不调整为 MerRec 纯文本路线。

补充证据：[Amazon 官方许可证讨论](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1) 中维护者明确表示无法为数据指定许可，同时认可非商业学术论文使用。后者不是一般数据授权，仍不满足本项目的明确许可门槛。

新增优先候选 C：[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)。官方规则搜索摘录明确包含非商业、学术研究与教育用途：[规则](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules)。任务基于历史交易与商品/客户 metadata，技术上适合图文商品推荐。官方下载入口要求登录并接受竞赛规则。完整规则/文件页在网页读取中为空，浏览器请求超时，因此尚未完成全部条款、下载大小与实际字段核验；不能以搜索摘录替代完整授权审阅。需要用户确认能否通过 Kaggle 官方入口取得数据，不代替用户接受条款，不索取令牌或密码。

下一步优先完整核实 C：articles/transactions 与商品图片按 article_id 对齐（需实测），ID作为字符串保留前导零；交易标签可支持下一商品购买推荐，但无曝光分母，不自动构成CTR。公开预期文件名/关系在下载后验证，不声称已经读取。只按官方授权下载必要 CSV 与少量对应图片，不下载全图片包或使用未经核实的镜像。当前尚未完成统一数据集选择，真实数据训练/评测/推理闭环尚未开始。

用户后续明确无 Kaggle 账号，优先无需登录的数据源；C 不再是当前主路线，不要求注册账号。

无需登录的补充核查：POG 官方 README 有 item_id/cate_id/pic_url/title 与用户商品序列，未见明确数据许可且示例不含时间；暂不采用。ABO 官方 CC BY 4.0、商品图文明确，但没有用户交互，不能单独满足推荐任务，不将其与无可靠联接/授权的行为数据强行拼接。VOGUE 官方仓库包含20名参与者、60段人类对话、商品图文与真实偏好评分，标注许可区分CC BY评分和研究限定图文；规模很小且不是电商交易日志，暂只作为微型真实偏好验证候选，完整 LICENSE、数据关联与适用任务仍需核查，不能替代正式规模实验。来源：https://github.com/wenyuer/POG 、https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html 、https://github.com/D3Mlab/vogue 。

VOGUE LICENSE.txt 已进一步读取：对话/调查/标注 CC BY 4.0；商品图文限定非商业研究学术用途，不得脱离数据集单独再分发图片；版权仍属于原持有人。这是发布方明示的使用条件，不构成对第三方权利的保证。可考虑只在本机研究使用其完整数据，先实际核对评分与商品图文的交集；任何正式任务调整须用户参与决定。

后续实际访问：官方GitHub文件树共210个文件、约40.5MB；已固定tree SHA并记录 reports/vogue_manifest.json。成功读取seeker_ratings.csv的60行及participant_id/conversation_id/item_1…item_36等字段。下载LICENSE本地副本、profile ratings和商品metadata样本超时（网页读取许可此前成功），完整图文联接尚未完成；失败记录见 reports/vogue_access.json，评分审计见 reports/vogue_ratings_audit.json。此数据为情境偏好调查，不应转换成有时间顺序的购买史；若使用，应采用参与者/商品分组评测，不伪造时间切分。尚未启动真实训练或服务。

## 评测协议草案（数据确定后锁定）

- 全局时间截点 train/validation/test；同时间戳事件成组，不随机拆散。历史必须严格早于目标。
- 训练统计、词表、负采样分布仅从训练数据拟合。商品首次观测只作为可用时间代理，明确缺少真实上架时间。
- 去重用户商品事件并保留规则；变体按 parent_asin 聚合。相同图片做哈希/近重复审计，不把推荐中正常的同商品跨期出现全部删除。冷商品实验单独按商品/近重复组隔离。
- 全量候选主评测限定为时点可用目录，剔除历史已见商品时同样处理目标与指标分母；不可评目标、冷用户、冷商品与覆盖率单独报告，不能静默过滤。
- 评论任务预测后续被评论商品或明确的高评分偏好，不预测点击率。未交互商品仅是采样未观测样本，不等于真实负反馈。
- 召回 Recall@K/NDCG@K/MRR；排序报告固定候选重排和端到端指标。记录候选生成版本、K、检索是否精确，以及目标不在候选的比例。采样候选指标另命名，不能与全量横比。
- 排序训练候选用时间内或折外召回构造，避免在同样本上过拟合召回后给排序喂不真实候选。困难负例只来自训练允许的候选，处理重复正例。

## 分阶段实施与验收

0. 可行性：核实许可/数据联接/历史覆盖，选小样本；读取 → 真实模型1 epoch → 全量小目录评测 → 单用户推理。保留数据哈希、依赖锁、权重、耗时与峰值内存；缺一项不算通过。不以随机张量探针代替。

1. 数据与基线：解决“推荐什么”的任务定义。输入商品目录/历史，输出候选与得分。流式清洗和全局时间切分；Popularity、训练期拟合 TF-IDF 历史聚合、冻结图文均值基线、简单 MLP 或 BPR 排序。选择少量基线便于归因。验收：泄漏检查、固定候选协议、真实训练与复现命令齐全。

2. 表示与召回：输入图/文及历史，输出128维用户商品向量；冻结小编码器，先训练投影/融合与用户塔，InfoNCE 或采样 softmax，处理 batch 内重复商品/已知正例。验收：文本/图片/图文对照，训练可恢复，验证集选择配置，测试集不调参。若可承担，再试小编码器部分解冻；不预先承诺大 VLM PEFT。

3. 排序与改进：固定召回候选，预测偏好排序分数；优先一个 MLP，历史确有价值再选 DIN。待用户提出/讨论假设后，研究缺图鲁棒融合或困难负采样，保留计算成本相近对照。验收：消融、失败记录、候选漏召回分析；条件允许3个种子，只报告实际运行数。

4. 部署：批量编码 → 精确向量检索 → 重排 → localhost API。先 NumPy/torch 精确检索，小目录无需提前加复杂 ANN；扩大后比较 ANN 召回损失。模型/词表/索引/配置的哈希版本一致，未知用户有显式回退。验收：离线服务相同输入得分一致；明确硬件、目录规模、batch与并发，测 warm/cold p50/p95/p99、吞吐、错误率、内存和索引字节，简洁演示。

每阶段开始先讲问题/输入输出/选择/验收，完成后沿真实代码讲张量、前向、损失、梯度、更新与指标，再提理解检查。依赖最终按实际可运行环境锁定；现在不编写成功实验、最终性能或简历收益数字。
