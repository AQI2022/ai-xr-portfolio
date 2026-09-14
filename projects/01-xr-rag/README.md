# XR 技术知识助手

把 XR 文档解析为带来源定位的片段，再检索与回答。核心文件：`ai_xr/rag.py`、`ai_xr/store.py`、`ai_xr/api.py`。

## 实现

PDF 保留原始页码；DOCX 读取段落与表格，但不虚构 DOCX 页码。UTF-8 TXT/MD 同样支持。按 700 字符切块、120 字符重叠，保存原文起止位置。文档名和内容哈希使重复上传幂等；同名但不同内容视为新文档，不暗中覆盖旧资料。

数据保存在 SQLite；向量矩阵在进程内加载，用文档签名触发重建。这是小规模精确向量检索，并非使用独立向量数据库或近似最近邻索引。默认 HashingVectorizer 是词法特征，不冒充 Embedding 模型；配置 BGE 后使用归一化语义向量。

BM25 与向量结果用 `1/(60+rank)` 融合，可选 CrossEncoder 重新排序。默认 top-k=4，返回标题、PDF 页码、字符区间和引用 ID。所有提示词把检索资料当作数据，不能用文档指令改变工具权限。

## 使用与评测

运行根目录 README 的服务，上传资料后调用 `POST /rag/query`。`/rag/stream` 是“处理阶段 + 最终结果”的 SSE，不是逐 token 模型流。

```bash
python experiments/evaluate_rag.py
python experiments/evaluate_rag.py --embedding models/bge-small-zh --reranker models/ms-marco-reranker
```

评测采用 12 条手工问题对 6 篇自编资料的文档级检索；只测命中和排名，不测复杂业务 QA。公开 `evidence/rag-lexical.json` 与 `rag-dense-reranked.json`，不要把两种模式不同时段的时延差当成严谨性能对照。

本地 0.5B 模型的生成曾缺引用并出现偏题。现对引用 ID 做结构校验，不合格则退回原文摘录，返回 `evidence_fallback=true`。合法引用仍不保证每句话被原文支持；后续应增加句级证据一致性、无答案问题与独立人工评审。

## 面试讲解

为什么不用大型向量数据库？当前资料少，精确检索容易复现和排错；规模变大后要持久化向量、增量索引、租户过滤和分离 embedding 服务。为什么重排未必更好？当前重排器主要面向英文，中文小测试集和饱和基线无法支持效果提升结论，应按语言选择模型并扩大未见测试集。
