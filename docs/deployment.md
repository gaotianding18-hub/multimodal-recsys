# 本机部署与复现

执行目录为项目根目录。本版仅用于非商业研究，不公开部署、不单独发布原始商品图片。没有云资源。

## 环境

Python 3.12.14，Apple M5/24GB，CPU训练和CPU服务（MPS虽可用，未用于正式实验）。

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-local.lock.txt
```

## 数据

data/vogue为官方VOGUE固定revision本地副本，数据未进入Git。下载图片：

```bash
.venv/bin/python scripts/download_vogue_data.py
.venv/bin/python scripts/prepare_vogue_ratings.py
.venv/bin/python scripts/audit_vogue.py
```

文字/CSV也必须使用reports/vogue_manifest.json指定版本：LICENSE.txt、data/fashion_profiles/ratings.csv、data/conversation_trials/item_ratings/seeker_ratings.csv、data/conversation_trials/scenarios.json、data/metadata/item_1.json…item_36.json。下载失败不得运行训练或伪造替代。完整数据获取入口见README；当前本机副本已齐全。

## 训练与检查

```bash
.venv/bin/python -m recsys.train
.venv/bin/python -m unittest discover -s tests -v
```

9组运行输出artifacts/runs，含model.pt、index.npy、features.npz、serving.json、manifest.json、metrics.json、training_log.json。reports/experiments.json保存基线和所有运行。默认服务使用验证NDCG@3最佳运行，商品图文双模态版本可以显式选multimodal-42。不根据测试结果选版本。

## 启动与停止

```bash
.venv/bin/python -m recsys.serve --port 8765
# 若要明确体验图文商品塔版本：先停止已有服务，再运行
.venv/bin/python -m recsys.serve --port 8765 --run multimodal-42
```

浏览器访问 http://127.0.0.1:8765 。默认已知用户s1…s10，场景1…6，输出3件商品。未知用户使用训练用户偏好均值回退。终端Ctrl+C停止；关闭承载服务的进程后需要重启。本机开发进程不等于系统守护服务，不保证应用退出后持续运行。

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/recommend -H 'Content-Type: application/json' -d '{"user_id":"s9","scenario":1,"k":3}'
.venv/bin/python scripts/benchmark_service.py
```

端口被占用时停止原进程或指定其他端口（性能脚本默认8765）。若文件校验失败，重新生成一致的模型/索引/metadata，不跳过校验。输出score是相对偏好分数，不是CTR。部分执行沙箱无法访问设备/监听端口，需本机终端或获准的宿主执行权限。
