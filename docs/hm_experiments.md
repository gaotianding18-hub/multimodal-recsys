> 2026-10-12：本文包含历史阶段记录。当前工程与实验结论以[实验总览](experiment_overview.md)为入口；原始reports保留，不因教学更新而覆盖。

# H&M 当前实验与失败分析

2026-10-11；本机M5/24GB，CPU。数据配置configs/hm.json实际驱动切分与用户选择。完整原始数据31,788,324行，但本轮选入100,000个训练活跃用户、1,342,361条窗口交易；212,658条按日basket训练查询、13,535条验证周查询、12,544条测试周查询。41,585个最近训练期出现商品构成全量候选，不使用采样评测。

验证目标商品覆盖43,014/45,880；测试35,783/41,779。目录外目标在Recall、NDCG、MAP分母中保留。模型无法推荐目录外商品。仅报告训练活跃用户子集，不代表所有用户/冷启动用户。日内无可靠时间顺序；同日输入不包含本日目标。历史可能含复购商品，允许再次推荐。

ID召回使用64维embedding、历史均值用户MLP、温度0.1的pairwise softplus/BPR；每batch对basket随机取一个正例。8个均匀未观测候选，排除用户训练期已知正例。mixed从32候选池保留4随机与4高分困难例，得分选择detach，只在选中候选上反传；增加采样/打分成本，不声称等计算量。3轮、batch256、AdamW lr0.002，各3种子。

实测随机负例test NDCG@12均值0.008768、种子总体标准差0.000177；mixed均值0.008511、标准差0.000414。平均训练+评测耗时分别17.11秒与20.77秒，未含数据准备或未来图文编码。困难负例没有稳定收益，不写成功提升。各epoch日志、权重、索引保留；汇总reports/hm_id_summary.json按验证NDCG选择random/seed43。标准差不是置信区间。

强基线：固定训练最后一周热门test NDCG@12=0.010472；最近历史复购优先、热门补齐=0.038473。两者明显优于ID网络。热门统计来自全体窗口训练用户，ID网络只用所选用户；这是使用额外训练群体信息的保守参照，不能称完全相同训练样本的受控消融。同日复购优先顺序采用ID确定性打破平局，不声称真实日内先后。基线复现scripts/evaluate_hm_baselines.py。

排序采用参考DIN思路的候选相关兴趣网络，附加历史是否出现、截断历史出现频次和训练期商品热度。冻结验证选出的召回模型，用完整允许商品中的均匀未观测负例训练pairwise softplus，不用同训练查询的拟合TopK生成排序训练池。训练期总体热度不是逐日因果快照，尚需改为按query截止日期构造。推理从召回Top100重排到12。

排序验证选第3轮；test NDCG@12=0.012795、Recall@12=0.016848、召回Recall@100=0.063554。排序改善了当前ID模型，但仍远低于复购基线。当前召回漏掉绝大多数目标，先改善候选而非堆排序模型。候选混合复购/近期热门可作为下一项基线设计，不提前算原创改进。

这些测试结果已经用于探索诊断，不称最终盲测。后续严格冻结实验计划，并增补预先约定的滚动时间窗口评测。大目录正式结果为ID/冻结CLIP文本实验；另已完成590商品图文匹配、条件购买召回及SmolVLM-500M生成式微调，分别见multimodal_experiments.md与generative_vlm.md，不与本节跨目录横比。无CTR/CVR、业务收益或企业生产经验。冻结主干+任务投影训练不等同于VLM2Vec主干微调。个人原创贡献未提前署名。

## 复现命令

项目目录依次执行：

```bash
.venv/bin/python scripts/audit_hm.py
.venv/bin/python scripts/prepare_hm.py
.venv/bin/python scripts/run_hm_experiments.py
.venv/bin/python -m scripts.evaluate_hm_baselines
.venv/bin/python -m recsys.hm_rank
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m recsys.hm_serve
```

run_hm_experiments跳过已有结果；主动重新实验需新建run标识或归档对应结果文件，不能把旧报告当作修改代码后新跑的结果。训练依赖requirements-hm.lock.txt。图片下载独立环境，见docs/kaggle_local_setup.md。

## 本机服务

http://127.0.0.1:8766/ 。旧8765仍是VOGUE。新服务实际使用H&M模型，30个匿名历史演示档案截止2020-09-15，支持POST /recommend指定history商品ID或demo用户；空历史回退训练热度，分数为偏好logit。启动核对权重SHA、索引重编码一致和排序模型匹配。未配置系统守护进程，不是长期生产部署。

30条测试输入的离线Top12与服务Engine逐项一致。HTTP warm测试5次预热、各100请求，并发1 p50=1.468ms/p95=4.983ms/441.2req/s；并发4 p50=2.764ms/p95=3.701ms/1382.7req/s。仅当前本机、41,585商品、CPU单torch线程、Top100重排与短请求负载；不外推线上SLA。报告reports/hm_consistency.json、hm_http_performance.json。

## 2026-10-11追加实验（以上为历史版本）

CLIP官方权重353,976,522字节，SHA256=40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af；冻结主干，官方分词器/模型源码MIT实际复用。MPS128件编码3.2686秒，125张图片可用；全目录41,585条文本编码129.0473秒，590张图片可用，图文完整训练未完成。结束时allocated约605MB、driver约1.62GB，仅结束时值，不称峰值。

文本任务训练：recsys/hm_multimodal.py，CLIP文本512维投影64维+ID64维，组合后归一化，历史均值用户MLP；所有basket正例加入去重候选，额外256随机商品，屏蔽用户训练已知正例但保留本日正例；多正例softmax，温度0.1，AdamW lr0.001，batch256、3轮、seed42。训练212,658查询，46.33秒（含每轮验证与最终测试）。验证选择第3轮，test Recall0.011497/NDCG0.007029/MAP0.003862。仅文本分支使用该特征快照；其中1.42%有图不影响文本模型，不能写成图文实验。报告reports/hm_mm_text_42.json是当前历史第一次运行；后续新运行按特征SHA命名，避免覆盖旧快照。

冻结CLIP文本历史均值余弦基线test Recall0.006924/NDCG0.004180/MAP0.002416。任务训练高于该单种子基线，但未超过ID三种子均值、近期热门和强复购，不宣称稳定提升。不同ID和对比模型损失/候选机制不同，不能把两者差异归因于文本唯一因素；正式模态对比需同架构、相同训练配置与特征快照。

多路候选诊断：ID100+历史最多30+近期热门100去重，候选test覆盖0.149469，原神经100候选0.063554。新排序去除总训练期热度特征，历史非本日目标/随机非本日目标混合未观测候选，3轮seed42，验证选第1轮；test NDCG0.007150，神经候选限定0.007165。覆盖改善未转化为最终排序改善；两个排序版本的训练采样/热度输入同时改变，不能将效果差异全部归因于候选混合。报告reports/hm_hybrid_id-random-43_42.json。

图片下载最初100验证成功后扩至目录；八线程触发官方429，产生无效失败重试，已停止且失败记录保留。实际590张（134,365,293字节），全部解码通过，reports/hm_image_audit.json保存SHA/相同dHash分组。相同dHash只是相似候选，不等于证明重复。时间推荐允许同商品跨时间出现；未做商品隔离冷启动验证。下载器已修复同名.jpg内容实际ZIP的情况，默认一线程、2秒请求间隔、遇429立即停止。不规避限流。

## 新服务验证

当前服务默认复购+近期热门，ID模型作为实验选项；图片只展示实际已取得文件；空/全未知历史回退近期热门。新版本7eaafc6f122f两策略/图片路由通过。本机M5/24GB、CPU torch1线程、30个历史、Top12、5预热、每策略每并发100次：复购并发1 p50=0.972ms/p95=1.175ms，993.34req/s；ID并发1 p50=1.089ms/p95=1.155ms，907.84req/s。并发4详见reports/hm_http_7eaafc6f122f.json。短warm负载，未测服务器峰值内存，不是线上SLA。旧性能报告仅适用于对应版本。

## 后续正式流水线

图片齐全后，先核查至少95%有效覆盖，再使用：

```bash
.venv/bin/python scripts/extract_hm_selected_images.py /absolute/path/to/official-archive.zip
.venv/bin/python scripts/complete_hm_upgrade.py --images-ready
```

流水线计划：图像哈希审计→断点CLIP全图文编码→text/image/mean/gate各3种子→验证选择召回→排序3种子→索引/版本/30条离线服务一致性验证→release发布到本机目录。image分支缺图时使用文本回退，报告必须注明；不是100%纯图像消融。当前未执行完，reports/hm_upgrade_progress.json记录图片阶段失败。正式HTTP重启、负载测试及滚动窗口仍需完成。

## 后续完善：同样本基线、文本稳定性与训练预算

文本3轮seed42/43/44已全部真实完成，统一特征SHA=08e0dec9…，NDCG@12分别0.007029/0.007264/0.006522，均值0.006938、种子总体标准差0.000310。验证选择seed42。标准差不是统计置信区间。报告reports/hm_text_summary.json。

新增同训练查询目标计数基线，不使用其他训练用户；query_popularity test NDCG0.008210，query_recent_popularity0.010455，repeat_then_query_recent0.038391。计数是每日去重basket目标出现次数，不是购买件数，首次仅输入历史basket未追加到热度计数。同样本参照仍显著高于3轮文本；此前全训练群体基线的信息优势并不能解释整个差距。报告reports/hm_matched_baselines.json。

单种子训练预算诊断：text/seed42从头训练10轮，验证NDCG第三轮0.009754、第十轮0.015714，验证选择第十轮；test Recall0.018075/NDCG0.011314/MAP0.006617。更长预算有收益，但不能称等计算量的算法改进，未超过复购基线且尚未验证多种子稳定性。run-tag budget10使旧3轮权重、报告保留；同种子重跑并非承诺逐比特一致。

真实失败：十轮训练已保存权重/索引后，报告增加源码哈希时漏导入Path而失败。reports/hm_budget10_report_failure.json记录错误，原执行源码保存在该run目录。修复后使用recover_hm_mm_report.py从保存权重恢复，并复核保存checkpoint验证NDCG与最佳epoch日志一致。原完整耗时未保留，报告seconds=null，仅记录恢复评测耗时，不重构虚假训练时间。另1024查询/1轮report-smoke确认修复后的完整报告写出链路正常，烟雾指标不用于效果结论。

服务加入目录ID顺序/配置协议拒绝启动检查、仅方法/状态/耗时/版本的轮转日志。12项测试通过；repeat/id各30输入离线服务Top12一致；HTTP非法参数六项返回400，空历史成功回退。55be40cefea2版本每策略每并发100请求无错误，完整负载条件在对应reports/hm_http_55be40cefea2.json，旧版本数字不代表新版本。

文本budget10的兴趣排序3轮，验证选第2轮：test NDCG0.007158/Recall0.012924，混合候选覆盖0.163122；神经候选限定NDCG0.006743。排序仍低于直接文本召回0.011314，不作为成功改进。该text-only模型仅作为本机实验选项发布，scope=text_only_experimental，默认复购基线保留。

正式加载前semantic30条离线/服务结果一致，回退/未知历史通过；独立校验进程峰值RSS1,685,585,920字节，是macOS进程内存，不是显存，也不是HTTP服务实测峰值。warm引擎单线程100调用p50=1.464ms/p95=1.667ms。

b6220dc58817服务版HTTP三策略、并发1/4、各100请求无错误；semantic并发1 p50=1.782ms/p95=1.956ms/544.63req/s，并发4 p50=5.982ms/p95=7.221ms/661.85req/s。负载为30匿名演示历史、Top12、5预热，CPU torch1线程、M5/24GB localhost；不等同生产流量。reports/hm_http_b6220dc58817.json。
