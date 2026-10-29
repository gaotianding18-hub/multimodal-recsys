# 第三方基础与署名

VOGUE：David Guo、Minqi Sun、Yilun Jiang、Jiazhou Liang、Scott Sanner，VOGUE: A Multimodal Dataset for Conversational Recommendation in Fashion，UMAP 2026。来源https://github.com/D3Mlab/vogue 。固定revision 68edb8e1ac43f3718844e00a0f40f4f10173b24b。对话/调查CC BY 4.0；商品图文限定非商业研究与学术用途，不得单独再分发图片。完整官方使用条件保存在data/vogue/LICENSE.txt，原权利人保留版权。本项目不公开原始图片或数据副本。

TorchEasyRec：Alibaba PAI Team，https://github.com/alibaba/TorchEasyRec ，Apache-2.0，参考推荐双塔/排序/特征一致性设计。VLM2Vec：TIGER-AI-Lab，https://github.com/TIGER-AI-Lab/VLM2Vec ，Apache-2.0，参考多模态对比学习思想。本版recsys代码为助手在本项目实现，不宣称原创框架；官方代码核查快照在reports中，附上游许可来源。数据与模型权重许可不能从代码许可推导。

PyTorch、NumPy、Pillow等依赖遵循各自许可证；版本清单requirements-local.lock.txt。未来引入预训练权重需另外核实其许可。
# CLIP 实际源码复用（升级阶段）

third_party/clip/model.py与simple_tokenizer.py来自OpenAI CLIP官方仓库，MIT许可证保留于third_party/clip/LICENSE。读取时Git blob分别为232b7792eb97440642547bd462cf128df9243933与0a66286b7d5019c6e221932a813768038f839c91。官方来源https://github.com/openai/CLIP 。这些文件不是本项目原创代码。官方发布权重与词表另记录下载哈希；实际编码、CLIP内部LoRA训练已完成，核查见reports/clip_source_audit.json与reports/clip_lora_summary.json。

## SmolVLM 生成式扩展

Hugging Face HuggingFaceTB/SmolVLM-500M-Instruct，模型卡声明Apache-2.0；固定revision a7da5b986cb59b408707209984f360a5f4ad7e47。源模型卡保存在本机artifacts/backbones/smolvlm-500m/README.md，公开URL：https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct 。源码由已固定版本transformers的Idefics3实现提供，没有复制官方微调notebook。模型权重不提交Git。模型卡、训练数据及基础模型的许可/限制继续适用，不宣称独立原创基础模型。
