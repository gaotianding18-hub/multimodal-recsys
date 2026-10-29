# 当前实验总览与证据索引

更新：2026-10-12（Asia/Shanghai）。根据当前训练/评测/服务代码、实际JSON报告和本机权重整理，不新增实验、不重写历史报告。旧文档的待实施叙述不能代表当前工程。

## 已完成范围与核验方式

已实现并运行：H&M大目录ID与CLIP文本召回、兴趣排序、多路候选；590商品图文匹配CLIP内部LoRA及27次购买召回实验；SmolVLM-500M生成式答案掩码SFT与视觉Token对照；本机推荐/属性API及版本化一致性检查。

本轮阅读数据准备、ID/多模态召回、排序、CLIP LoRA、正式VLM训练、Token对照与服务代码；读取reports与原始CSV表头/样例，确认best.pt/model.pt存在。VLM best.pt SHA与vlm_sft_42.json一致。没有重新扫描31,788,324行交易或重跑训练/测试，以下数值为已有报告，不是本轮新测。8766/8767健康读取均返回URLError，当前会话无法确认服务实时可用，未自动重启；历史HTTP报告仍是对应版本的证据。

## 数据与任务契约

H&M官方竞赛CSV，用户此前接受规则；按现有授权仅本机非商业学习研究，不再分发原数据。许可记录见data_hm.md与provenance.md。交易字段为t_dat/customer_id/article_id/price/sales_channel_id；当前推荐输入使用过去购买商品ID，不使用客户静态属性、交易价格或渠道作为模型特征。article_id保持10位字符串和前导零。

reports/hm_audit.json：31,788,324交易、105,542商品、1,371,980客户；未知商品/用户交易行为0，缺描述416商品。代码检查ID长度/数字与日期长度，并非完整日期解析或全部字段语义验证。重复交易可代表多次购买，原始行不机械删除；build_queries把同日购买去重为集合，是任务转换，不是断言重复行错误。静态目录/客户数据无历史版本。

reports/hm_preparation.json与configs/hm.json：训练2020-06-17至09-08，验证09-09至09-15，测试09-16至09-22；仅按训练记录选择100,000用户。1,342,361窗口交易、212,658每日训练查询、13,535验证/12,544测试周查询、41,585训练期候选。每日目标在创建查询后才追加历史，评测历史冻结于周初。测试周历史允许包含验证周已发生购买；候选目录固定为训练窗口观测商品。目录外目标保留为未命中，测试目标35,783/41,779在目录内。

已有590件官方图片，非随机取得；图文匹配按product_code/完全相同图片SHA/完全相同文本联合分226组，413/93/84商品。购买子集沿用时间切分，过滤为图像可用历史与目标，1794/150/124查询、全590候选、历史10件。此任务改变了目标与样本分母，不能与大目录指标横比，也不证明推荐商品冷启动。

## 实验一：大目录基线、ID与文本召回

代码：hm_train.py用Embedding与用户MLP，历史[B,30]→[B,30,64]→[B,64]，全目录[N,64]点积。随机8候选负例，mixed从32候选选4随机+4困难；softplus((s_neg-s_pos)/0.1)均值，AdamW更新。已知训练正例从负例排除，未观测不是明确负反馈。

hm_multimodal.py与advanced.py：冻结CLIP特征，512→64语义投影、ID64与语义64拼接后128→64，mask历史均值与MLP。batch全部正例加256随机候选去重，positive/allowed mask控制多正例log_softmax均值损失，温度0.1。训练更新任务头，不更新CLIP。验证NDCG@12选checkpoint，评测全41,585候选。

关键证据与结论：

- hm_id_random_42/43/44.json、hm_id_mixed_42/43/44.json：两种负采样三种子，困难负例没有稳定收益；详见hm_experiments.md。
- hm_matched_baselines.json：同212,658查询计数，test热门/近期热门/复购+近期热门NDCG@12约0.008210/0.010455/0.038391。
- hm_text_summary.json：3轮、batch256、lr0.001、seed42/43/44；test NDCG均值0.006938、种子总体std0.000310，明显低于强复购。冻结文本均值基线约0.004180，见hm_frozen_clip_text_baseline.json。
- hm_mm_text_42_08e0dec9-budget10.json：单seed42增至10轮，验证选第10轮，test NDCG0.011314，仍低于复购；增加预算不构成等计算量算法改进。

十轮报告曾因缺Path导入失败，权重/日志保留，随后恢复并复核验证日志。hm_budget10_report_failure.json保留失败；总时间缺失，seconds=null，不能用恢复评测耗时代替训练总时间。

## 实验二：兴趣排序与多路候选

代码：advanced.py::InterestRanker利用历史/候选/差/乘积形成候选相关注意力；hm_rank.py输出偏好logit，pairwise softplus损失。hm_hybrid.py冻结召回表示，合并神经Top100、最多30历史与近期热门100；新版总体热度特征置0，使用历史出现等特征，采样未观测干扰商品而非真实曝光负例。

hm_hybrid_id-random-43_42.json：test候选覆盖约14.95%，原神经候选约6.36%，覆盖提高但NDCG下降；采样与特征也变化，不能唯一归因候选融合。

hm_hybrid_mm-text-42-08e0dec9-budget10_42.json：3轮、验证选第2轮，test混合候选覆盖0.163122，端到端NDCG0.007158，低于直接文本召回0.011314。排序已真实运行，但不是效果成功案例。

## 实验三：CLIP内部LoRA图文匹配

代码：clip_lora.py、train_clip_lora.py。官方OpenAI CLIP ViT-B/32，MIT源码实际复用。图片[B,3,224,224]、文本[B,77]→两塔[B,512]→相似度[B,B]，双向多正例softmax损失，温度0.07。完全相同图片或文本为正匹配。

最后一层视觉与文本Transformer的MLP c_fc/c_proj注入rank4/alpha8 LoRA，51,200参数；基础冻结，AdamW lr1e-4、batch16、3轮、MPS、seed42/43/44，按验证双向R@1均值选权重。

clip_lora_summary.json：84测试商品，冻结图→文R@1 0.607143，LoRA三种子均值0.690476；文→图0.607143→0.615079，seed std0.014848。基础SHA不变、两塔非零梯度；checkpoint抽样重编码一致见clip_lora_checkpoint_check.json。训练约19—23秒/次；结束内存不是峰值。小样本匹配结果不等于购买收益或统计显著。

## 实验四：图文购买召回及LoRA迁移

代码：train_visual_retrieval.py复用MultimodalRetrieval和多正例损失；CPU、64维、batch64、10轮、AdamW lr0.001、额外128随机候选，验证NDCG选权重，测试全590候选。

visual_summary.json：冻结/LoRA缓存各text/image/mean/gate四模式×3召回种子，加3个ID基线，共27次训练。LoRA特征固定适配器seed42，三个召回种子不是适配器交叉实验；text/image均含ID，ID基线历史架构不同。

test NDCG@12均值：ID0.275858；冻结text/image/mean/gate为0.291401/0.290838/0.291654/0.292176；LoRA为0.285951/0.293219/0.289512/0.290878；复购0.307877。融合和LoRA推荐迁移均无稳定收益。验证选冻结gate seed43，test0.291888；visual服务使用冻结特征，不使用LoRA特征。

## 实验五：生成式VLM答案掩码SFT

代码：prepare_vlm_attributes.py、finetune_vlm_attributes.py、vlm_metrics.py。train_generative_vlm.py只对应4步pilot，不代表正式训练入口。

SmolVLM-500M-Instruct，Apache-2.0，固定revision与官方权重SHA。同413/93/84商品分组；仅图片+固定指令输入，类别/颜色目录字段作为JSON弱标签答案，不把答案放入问题。关闭图像切片，longest_edge512；标签提示前缀设-100，答案与EOS参与模型内部因果语言交叉熵。语言注意力q/v注入rank4/alpha8 LoRA，共409,600参数，基础冻结，batch1、3轮、AdamW lr1e-4、MPS、seed42。

vlm_sft_42.json：非零梯度/适配器变化、基础SHA不变，按验证类别/颜色宏召回均值选第3轮。84测试商品JSON有效率0.9048→1.0000，类别一致率0.2024→0.4286，颜色0.4524→0.6429，联合0.1071→0.2857。总638.31秒包含基线/验证/测试生成，不是纯训练时间。

这是目录弱标签一致率，不是人工视觉准确率、推荐收益或大规模指令微调。单种子、非随机商品覆盖；25件训练未见目标类别一致率0.08，59件已见类别0.5763，见本机error_analysis.json。输出属性尚未接入推荐特征。

## 实验六：视觉Token保留对照

代码：benchmark_vlm_tokens.py。固定选中适配器，93验证商品、batch1 Float32 MPS、greedy最多60新token；完整视觉编码器和连接器先运行，再对64视觉向量保留uniform32/norm32/uniform16，并同步压缩语言输入中的视觉占位token。

vlm_token_benchmark.json：full/uniform32/norm32/uniform16联合目录一致率0.3011/0.3118/0.3441/0.2688；p50约471.1/461.9/471.7/486.6ms。norm32未实际加速，16个token损失质量。固定顺序单次、输出长度不同，不能据此宣称稳定质量提升或速度收益。计时含视觉编码与生成，排除读图/预处理；完整视觉编码成本没有减少，结束内存不是峰值。均匀/范数方法为既有基线，无用户原创剪枝结论。

## 部署与一致性证据

hm_serve.py：本机8766，repeat/id/semantic/visual；semantic为41,585文本实验，visual为590图文子集，默认强复购。检查权重/索引/配置/目录顺序，未知历史回退，非法参数拒绝。visual_service_consistency.json：124查询Top12一致、版本fd16920e3efd；对应hm_http_fd16920e3efd.json四策略、并发1/4、各100请求无错误。短warm本机负载不等于生产SLA。

vlm_serve.py：本机8767、标准64视觉token，只接受已知商品ID，校验基础/适配器/数据提示SHA。vlm_http_check.json实际记录版本20e00e7b94ee，3测试商品离线/HTTP生成一致及非法请求400。旧PROJECT_STATE记的a05b4fc20f8f不能替代该报告版本；也不能据该报告推断当前运行版本。没有生成式API并发压力或峰值资源验证。

## 已完成、未完成与个人贡献

研究原型的训练—评测—保存—本机推理链路已完成。尚未完成：全41,585商品图文覆盖、新预声明时间窗口泛化、公平同容量/预算融合对照、稳定Token效率重复验证、后续代码复现及新增改进。它们是可选研究深化或实践，不把现有工程说成尚未训练。

下载429、完整升级流水线failed为历史失败；用户拒绝35GB整包，当前不继续该下载路线。复杂模型未稳定超过复购、排序退化与迁移失败均保留。测试已探索性查看，不称盲测。没有曝光CTR/CVR、线上A/B或企业生产结果。

本项目由用户个人独立负责并完成，使用自行编写prompt组织AI工具辅助实现和实验。课程概念主线已学习完成，贡献归属见provenance.md，问答与纠正过程见learning.md；后续新增代码实践按实际记录追加。

## 使用已有证据与后续复现

先读本页→任务对应代码→对应reports→权重与日志。详细运行入口见hm_deployment.md、multimodal_experiments.md、generative_vlm.md；复现命令可能重建数据、覆盖同名产物，教学阶段不直接整段执行。重训前预声明目的、预算、保存目录与验收标准；优先小规模独立练习，保留全部历史结果。
