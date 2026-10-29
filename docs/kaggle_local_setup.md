# 本机图片下载凭据

用户已选择本人配置Kaggle API凭据。下载器独立环境为.venv-kaggle，固定客户端kaggle==1.7.4.5，不改变训练环境。

1. 登录Kaggle，打开本人Settings中的API区域。
2. 当前下载器使用username/key格式：若页面提供Legacy API Credentials / Create Legacy API Key，生成并下载kaggle.json。若只有新式token选项，不把token改名成JSON；需升级下载适配器。
3. 在本机终端执行`mkdir -p ~/.kaggle`，把本人下载的kaggle.json放入该目录，并执行`chmod 600 ~/.kaggle/kaggle.json`。
4. 不把凭据发到聊天或放进项目数据、Git仓库。配置好后只告知已配置，助手运行官方接口校验。

项目目录内执行`.venv-kaggle/bin/python scripts/download_hm_images.py --limit 100`进行接口与图像验证；验证通过再使用`--limit 0`取得整个所选目录。现有文件跳过；每次失败留档，404不伪造图片，401/403停止。初始100张只是下载烟雾检查，不能用有图的前100商品代表完整推荐实验。

图片通过训练期商品ID对齐，保留images/前三位ID/完整ID.jpg结构。完整像素解码、近重复检查和图文编码仍需后续执行。


2026-10-11已验证凭据和比赛权限：官方图像成功取得，随后逐图请求触发429，当前已停止。新版默认workers1、request-interval2秒、429立即停止，先等待恢复，不加线程规避。整包方式尚在等待用户的磁盘空间决策；脚本extract_hm_selected_images.py仅负责从已授权官方包中提取目录图片，不代替接受规则。
