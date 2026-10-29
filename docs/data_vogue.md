# VOGUE数据准备：教学核验候选

用途：本机非商业研究与字段教学；正式项目数据尚未决定，不将此小调查数据冒充交易推荐规模。

来源：https://github.com/D3Mlab/vogue ，固定revision：68edb8e1ac43f3718844e00a0f40f4f10173b24b。许可证本地副本data/vogue/LICENSE.txt：对话/调查CC BY 4.0；商品图文限定非商业研究、禁止脱离数据集单独分发图片。版权归原权利人。原数据不进入Git。

## 已运行的数据处理

scripts/prepare_vogue_ratings.py读取seeker_ratings.csv。原表一行对应一次对话情境下的评分，item_1…item_36为商品评分列；-1为缺失，1…5为观察到的评分。

脚本输出data/processed/vogue_observed_ratings.jsonl：每行含user_id、conversation_id、scenario、catalogue、item_id、rating、label_source。保留情境，不平均同用户商品不同情境的分数；不虚构时间、点击或购买，不输出隐式历史。

实际输出720条，所有评分1…5，(user_id,conversation_id,item_id)无重复；涉及10名Seeker、36件商品。商品1有20条情境评分，不等于20名用户或20次购买。

商品文字通过metadata/item_1.json的id=1关联item_id=1；商品名为Wantdo Men's Cotton Lightweight Jacket Military Jacket Front Zip Jacket。product_description为null，可选择真实product_name/about_product文本并记录规则，不伪造描述。禁止将product_rating或reviews汇总当成用户标签；后者是摘要，不用于此次图文输入。

商品图预期对应metadata/item_1.png，是否成功补齐及完整联接覆盖率以reports/vogue_data_audit.json实际输出为准。文件存在只是初步核验，不意味着全部内容和协议完成。

## 复现命令

```bash
.venv/bin/python scripts/prepare_vogue_ratings.py
.venv/bin/python scripts/audit_vogue.py
```

尚未做训练切分、训练或效果评测。情境评分需要分组协议，不能按虚构购买时间切分。正式选用/缩小任务范围前须用户参与决定。
