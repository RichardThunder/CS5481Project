# RAG 系统改进指南 - 针对学术论文

## 🎯 当前问题

✗ **文档太少**: 只有 2 篇论文
✗ **覆盖面窄**: 只测试单一主题
✗ **缺乏深度**: 没有跨文档推理测试
✗ **问题单一**: 缺少不同难度和类型的问题

---

## 🚀 完整改进方案

### 阶段 1: 扩充文档集合

#### 方案 A: 手动下载论文 (快速)

访问 [arXiv.org](https://arxiv.org) 下载以下论文:

**RAG 相关 (必须):**
- `2005.11401` - Retrieval-Augmented Generation (原始论文)
- `2310.11511` - Self-RAG
- `2312.10997` - RAFT
- `2404.16130` - Graph RAG

**LLM 基础 (推荐):**
- `2307.09288` - Llama 2
- `2310.06825` - Mistral 7B
- `2203.02155` - InstructGPT

**知识图谱 (推荐):**
- `2308.08998` - Knowledge Graphs for LLM
- `1609.02907` - Graph Convolutional Networks

**评估方法 (推荐):**
- `2309.15217` - RAGAS Paper
- `2307.03109` - LLM Evaluation Survey

#### 方案 B: 使用脚本自动下载

```bash
# 安装依赖
pip install arxiv

# 运行下载脚本
python scripts/download_papers.py
```

**推荐的文档组织结构:**

```
documents/
├── rag/           # RAG 相关论文 (核心)
├── llm/           # LLM 基础论文
├── knowledge_graph/  # 知识图谱论文
└── evaluation/    # 评估方法论文
```

---

### 阶段 2: 改进测试策略

#### 新增 8 种问题类型

| 类型 | 说明 | 示例 | 难度 |
|------|------|------|------|
| **单文档事实** | 提取具体信息 | "GraphRAG 使用什么数据库?" | Easy |
| **单文档概念** | 理解核心概念 | "为什么图检索重要?" | Medium |
| **跨文档对比** | 比较不同方法 | "GraphRAG vs 标准 RAG?" | Medium-Hard |
| **跨文档综合** | 综合多篇论文 | "如何改进 RAG 系统?" | Hard |
| **方法论深度** | 技术细节理解 | "混合查询如何实现?" | Hard |
| **应用场景** | 实际应用能力 | "如何应用于金融?" | Medium |
| **批判性分析** | 识别假设和局限 | "GraphRAG 的局限是什么?" | Hard |
| **边界情况** | 处理特殊情况 | "如何处理矛盾信息?" | Hard |

#### 难度分级

```python
Easy (简单):
- 直接事实提取
- 单一信息点
- 明确答案

Medium (中等):
- 需要理解概念
- 需要简单推理
- 可能跨多个段落

Hard (困难):
- 需要跨文档综合
- 需要深度推理
- 需要批判性思考
```

---

### 阶段 3: 运行改进后的测试

#### 步骤 1: 扩充文档

```bash
# 下载更多论文 (推荐至少 10-15 篇)
python scripts/download_papers.py

# 检查文档
ls -R documents/

# 重新摄取 (重要!)
rm -rf chroma_db/
python ingest_documents.py
```

#### 步骤 2: 生成综合测试集

```bash
# 使用改进的生成器 (包含 20+ 个问题,覆盖 8 种类型)
python src/test/generate_comprehensive_ragas_dataset.py
```

**预期输出:**
```
Total samples: 21

By Category:
  single_doc_factual       :  3
  single_doc_conceptual    :  3
  cross_doc_comparative    :  3
  cross_doc_synthesis      :  2
  methodological           :  2
  application              :  3
  critical_analysis        :  2
  edge_cases               :  2

By Difficulty:
  easy      :  3
  medium    :  8
  hard      : 10
```

#### 步骤 3: 运行评估

```bash
# 使用原有的评估脚本
python src/test/evaluate_academic_ragas.py
```

现在会看到更细致的分析:
- 总体表现
- 按问题类型的表现
- 按难度的表现
- 针对性改进建议

---

### 阶段 4: 迭代优化

#### 4.1 分析弱项

查看评估结果,识别哪些类型的问题表现差:

```
FACTUAL Questions:          0.85  ✓ 很好
CONCEPTUAL Questions:       0.72  ⚠ 需要改进
CROSS-DOC COMPARATIVE:      0.65  ⚠ 明显弱项
CROSS-DOC SYNTHESIS:        0.58  ⚠ 需要重点优化
```

#### 4.2 针对性优化

**如果跨文档推理弱 (< 0.65):**

```yaml
# config.yaml
retrieval:
  top_k: 6  # 增加检索数量
  search_type: "mmr"  # 使用 MMR 增加多样性
```

**如果概念理解弱 (< 0.70):**

```yaml
# config.yaml
chunking:
  chunk_size: 1500  # 增加 chunk 大小保留更多上下文
  chunk_overlap: 300  # 增加 overlap
```

**如果生成质量弱:**

```python
# 改进 prompt
prompt = """
You are an AI research assistant. Answer based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Use only information from the context
2. If information is incomplete, state that clearly
3. Cite specific parts of the context when possible
4. For comparison questions, contrast different approaches systematically

Answer:
"""
```

#### 4.3 重新测试

```bash
# 删除旧的向量数据库
rm -rf chroma_db/

# 重新摄取 (应用新的 chunking 配置)
python ingest_documents.py

# 重新生成测试集
python src/test/generate_comprehensive_ragas_dataset.py

# 重新评估
python src/test/evaluate_academic_ragas.py
```

#### 4.4 对比前后

记录每次优化的结果:

| 优化阶段 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Overall |
|---------|-------------|------------------|-------------------|----------------|---------|
| 初始 (2篇论文) | 0.75 | 0.72 | 0.78 | 0.70 | 0.74 |
| +10篇论文 | 0.78 | 0.75 | 0.82 | 0.74 | 0.77 |
| +优化 chunking | 0.80 | 0.77 | 0.84 | 0.76 | 0.79 |
| +改进 prompt | 0.83 | 0.80 | 0.84 | 0.77 | 0.81 |

---

## 📊 进阶改进技巧

### 1. Chunking 策略优化

**当前 (默认):**
```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200
```

**针对学术论文优化:**
```yaml
chunking:
  chunk_size: 1500  # 论文段落通常较长
  chunk_overlap: 300  # 保留更多上下文
  separators: ["\n\n", "\n", ". ", " ", ""]  # 优先在段落/句子边界切分
```

**自定义分块逻辑:**
```python
# 按论文章节分块
def chunk_by_sections(text):
    sections = re.split(r'\n\d+\.\s+[A-Z]', text)  # 按章节号分割
    return sections

# 保留论文元数据
metadata = {
    "title": "GraphRAG-Causal",
    "section": "3.2 Graph Retrieval",
    "arxiv_id": "2506.11600",
}
```

### 2. 混合检索策略

```python
# 结合多种检索方法
results_similarity = vector_store.similarity_search(query, k=5)
results_mmr = vector_store.max_marginal_relevance_search(query, k=5)

# 合并并去重
combined = merge_and_deduplicate(results_similarity, results_mmr)
```

### 3. Re-ranking

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 使用 LLM 对检索结果重新排序
compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)
```

### 4. 查询扩展

```python
# 使用 LLM 生成相关查询
expanded_queries = llm.invoke(f"""
Given the question: {query}
Generate 3 related questions that would help answer it.
""")

# 对每个查询检索,合并结果
all_results = []
for q in expanded_queries:
    results = vector_store.similarity_search(q, k=3)
    all_results.extend(results)
```

### 5. 自适应检索

```python
# 根据问题类型调整策略
if "compare" in query.lower():
    k = 8  # 对比问题需要更多上下文
    search_type = "mmr"  # 需要多样性
elif "what is" in query.lower():
    k = 3  # 定义问题需要精确答案
    search_type = "similarity"
```

---

## 🎯 评估基准

### 目标分数 (针对学术论文)

| 指标 | 最低目标 | 良好 | 优秀 |
|------|---------|------|------|
| **Faithfulness** | 0.75 | 0.80 | 0.85+ |
| **Answer Relevancy** | 0.70 | 0.75 | 0.80+ |
| **Context Precision** | 0.75 | 0.80 | 0.85+ |
| **Context Recall** | 0.65 | 0.75 | 0.80+ |
| **Overall** | 0.70 | 0.77 | 0.83+ |

### 不同问题类型的期望

| 问题类型 | 期望分数 | 原因 |
|---------|---------|------|
| 单文档事实 | 0.85+ | 直接提取,应该很准确 |
| 单文档概念 | 0.75+ | 需要理解,但信息集中 |
| 跨文档对比 | 0.70+ | 需要综合多个来源 |
| 跨文档综合 | 0.65+ | 最难,需要深度推理 |
| 边界情况 | 0.60+ | 测试系统极限 |

---

## 📈 持续改进流程

```
1. 基准测试
   ↓
2. 识别弱项 (哪类问题得分低?)
   ↓
3. 假设原因 (检索?生成?数据?)
   ↓
4. 针对性优化
   ↓
5. 重新测试
   ↓
6. 对比改进 → 如果不满意,回到步骤 2
   ↓
7. 记录最佳配置
```

### 实验记录模板

```markdown
## 实验 #3: 优化 Chunking

**日期**: 2025-01-15
**目标**: 提高跨文档推理能力
**变更**: chunk_size 1000→1500, overlap 200→300

**结果**:
- Context Recall: 0.70 → 0.76 ✓
- Cross-doc Synthesis: 0.58 → 0.65 ✓
- 单文档问题无显著变化

**结论**: 增大 chunk 对跨文档推理有明显帮助
**下一步**: 尝试 MMR 检索
```

---

## 🔧 快速检查清单

运行评估前确保:

- [ ] 至少有 10-15 篇论文
- [ ] 文档已重新摄取 (`python ingest_documents.py`)
- [ ] 测试集包含多种问题类型
- [ ] 测试集包含不同难度级别
- [ ] Ground truth 质量高 (详细、准确)
- [ ] 有跨文档推理的问题
- [ ] 有边界情况测试

---

## 💡 常见问题

### Q: 我的分数一直很低怎么办?

A: 按优先级检查:
1. **Ground Truth 质量** - 最常见问题!确保答案详细准确
2. **文档覆盖** - 是否有足够文档?
3. **Chunking** - 是否切分合理?
4. **Retrieval** - 检索是否精准?
5. **Generation** - Prompt 是否清晰?

### Q: 单文档问题好,跨文档问题差?

A: 这很正常!解决方案:
- 增加 `top_k` (4→6 or 8)
- 使用 MMR 检索增加多样性
- 增大 chunk size 保留更多上下文
- 考虑 re-ranking

### Q: 评估太慢怎么办?

A:
- 使用 `mistral-small` 而不是 `mistral-large`
- 减少测试集大小 (先用 10 个问题快速迭代)
- 批量化 API 调用

### Q: 如何判断是检索问题还是生成问题?

A: 分别查看指标:
- `Context Precision/Recall` 低 → 检索问题
- `Faithfulness` 低 → 生成问题 (幻觉)
- `Answer Relevancy` 低 → 可能两者都有问题

---

## 📚 参考资源

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Retrieval Guide](https://python.langchain.com/docs/modules/data_connection/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997)

---

祝你的 RAG 系统越来越好! 🚀
