# H&M本机复现与部署

这是研究演示，服务只绑定127.0.0.1。当前完整图文未完成；以PROJECT_STATE.md为准。以下命令在项目根目录执行。新机器需要本人按官方规则取得数据，原始数据和权重不在Git中。

## 环境

Python3.12.14，Apple Silicon macOS，本项目验证机器M5/24GB；依赖固定在requirements-hm.lock.txt。

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-hm.lock.txt
uv venv --python python3.12 .venv-kaggle
uv pip install --python .venv-kaggle/bin/python -r requirements-kaggle.lock.txt
```

TorchEasyRec/VLM2Vec完整环境未安装；本机独立实现对应推荐思想。CLIP源码、BPE与MIT许可在third_party/clip。MPS仅在宿主可用时执行；CPU路径也支持，不推测其他设备耗时。

## 数据、训练和编码

官方articles.csv、customers.csv、transactions_train.csv位于data/hm/raw。凭据只在本人~/.kaggle/kaggle.json，配置见kaggle_local_setup.md。不得将数据授权理解为任意再分发。

```bash
.venv/bin/python scripts/audit_hm.py
.venv/bin/python scripts/prepare_hm.py
.venv/bin/python scripts/run_hm_experiments.py
.venv/bin/python -m recsys.hm_rank
.venv/bin/python -m scripts.evaluate_hm_matched_baselines
.venv/bin/python -m scripts.download_clip_backbone
.venv/bin/python -m recsys.clip_features --limit 0 --device mps --batch-size 64
.venv/bin/python -m recsys.hm_multimodal --mode text --seed 42 --epochs 3
```

文本可先完成，缺图不得冒充完整图文。新超参数实验指定--run-tag避免覆盖同一特征/种子目录，例如--epochs 10 --run-tag budget10。报告包含trainer与训练查询哈希（旧历史报告不补造这些字段）。目录仍在编码时不要同时读取并训练该特征文件。

图片取得后执行scripts/audit_hm_images.py与scripts/complete_hm_upgrade.py --images-ready。它会拒绝有效图像覆盖低于95%的正式图文实验；自然缺图通过mask保留商品，不将其删除。完整官方压缩包的下载尚需用户确认空间方案；extract_hm_selected_images.py只提取训练目录图片，不展开所有文件。提取校验后再处理压缩包，不自动删除未经核验的原始资料。

## 服务和检查

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m scripts.verify_hm_current
.venv/bin/python -m recsys.hm_serve --port 8766
```

浏览器打开http://127.0.0.1:8766/。服务未配置系统守护进程；启动终端/进程退出后需要重新运行。不要重复启动同一端口，Ctrl+C停止前台服务。

GET /health返回模型/代码/目录版本及可用策略；GET /demo-users列出匿名演示用户。POST /recommend接受{"user_id":"demo-1","method":"repeat","k":12}，或{"history":["0108775015"],"method":"id","k":12}。history必须是字符串列表，最多200输入，实际取最后30个目录内商品；k为1..50整数。空/全未知历史回退近期热门。repeat得分是顺序分，id/semantic得分是偏好logit，都不是购买概率。

当前默认repeat强基线，id为实验。semantic已加载经验证的text-only实验release，增加训练预算的文本召回和兴趣排序；不是完整图文升级。目录ID顺序或训练配置漂移会拒绝启动；索引与模型编码不一致、release SHA不匹配也会拒绝。

请求日志位于artifacts/hm/service_logs/requests.jsonl，轮转最多三个5MiB文件，仅记录时间、版本、方法、状态、耗时。日志不记录用户ID、历史或请求正文；无公网权限管理、生产SLA或流量声明。

另一个终端运行：

```bash
.venv/bin/python -m scripts.benchmark_hm_http
```

基准按版本写报告，30个演示历史、Top12、5次预热，每策略/并发100次请求，比较并发1/4；不能把客户端内存视为服务器峰值内存，也不能把短localhost负载称线上性能。完整release须另运行verify_hm_release.py检查30条语义离线/服务结果。

## 新增真实图文子实验

现有图片无需追加下载，visual策略使用590候选、冻结CLIP图文门控召回；semantic仍为大目录文本。图文服务自动读取reports/visual_summary.json的验证选择，校验模型摘要并重编码索引核对。必须保留data/hm/visual、对应artifacts/hm/visual-frozen-gate-43及选择报告，不单独复制索引文件。CLIP LoRA训练/缓存及三种子报告为独立实验产物，当前服务未选LoRA特征。完整复现见multimodal_experiments.md。

当前已测试版本fd16920e3efd；新版本性能见reports/hm_http_fd16920e3efd.json，旧版本数字不自动沿用。本机HTTP短时测试不是生产压力测试。
