# RAGAS Evaluation for Academic Papers (AI 论文)

## 📋 概述

本目录包含专门为 AI 学术论文设计的 RAGAS 评估系统。

## 🎯 评估维度

### 1. 问题类型分类

| 类型 | 说明 | 示例 |
|------|------|------|
| **Factual** | 事实性问题 | "GraphRAG-Causal 的主要贡献是什么?" |
| **Conceptual** | 概念理解 | "图检索如何增强因果推理?" |
| **Methodology** | 方法论 | "混合查询机制如何实现?" |
| **Comparative** | 对比分析 | "GraphRAG 相比标准 RAG 有何优势?" |
| **Application** | 应用场景 | "如何将 GraphRAG 应用于金融新闻?" |

### 2. RAGAS 评估指标

| 指标 | 含义 | 评估内容 |
|------|------|----------|
| **Faithfulness** | 忠实度 | 答案是否基于检索的上下文(是否幻觉) |
| **Answer Relevancy** | 答案相关性 | 答案是否回答了问题 |
| **Context Precision** | 上下文精确度 | 检索的文档是否相关 |
| **Context Recall** | 上下文召回率 | 是否检索到所有相关信息 |

## 🚀 使用流程

### 步骤 1: 确保环境准备好

```bash
# 激活环境
conda activate rag_env

# 确保已安装依赖
pip install ragas datasets langchain-mistralai
```

### 步骤 2: 确保向量数据库已创建

```bash
# 检查是否有向量数据库
ls -lh chroma_db/

# 如果没有,运行摄取脚本
python ingest_documents.py
```

### 步骤 3: 生成 RAGAS 测试数据集

```bash
# 从项目根目录运行
python src/test/generate_academic_ragas_dataset.py
```

**输出:**
- 生成 `src/test/test_data/academic_ragas_dataset.json`
- 包含多种类型的问题(事实、概念、方法论等)
- 每个问题都有对应的标准答案(ground truth)

### 步骤 4: 运行 RAGAS 评估

```bash
python src/test/evaluate_academic_ragas.py
```

**输出:**
- 总体评估分数
- 按问题类型的分项评分
- 性能解释和改进建议
- 结果保存到 `src/test/test_data/academic_ragas_results.json`

## 📊 评分标准

| 分数范围 | 等级 | 说明 |
|---------|------|------|
| 0.8 - 1.0 | 优秀 | RAG 系统表现出色 |
| 0.6 - 0.8 | 良好 | 性能可靠,有提升空间 |
| 0.4 - 0.6 | 中等 | 需要优化检索或生成 |
| < 0.4 | 需改进 | 需要重大优化 |

## 🎓 针对学术论文的最佳实践

### 1. 设计高质量的 Ground Truth

**好的 Ground Truth:**
```python
ground_truth = """
GraphRAG-Causal combines graph-based retrieval with LLMs
to enhance causal reasoning by: (1) structuring news as
causal graphs, (2) using hybrid semantic + graph queries,
and (3) employing XML-based prompting for few-shot learning.
"""
```

**不好的 Ground Truth:**
```python
ground_truth = "It's a framework for causal reasoning."  # 太简略
```

### 2. 覆盖论文的不同部分

- ✅ Abstract/Introduction (动机和贡献)
- ✅ Related Work (与其他方法的对比)
- ✅ Methodology (技术细节)
- ✅ Experiments (结果和分析)
- ✅ Limitations (局限性)
- ✅ Future Work (应用场景)

### 3. 测试不同难度级别

**简单:** "论文使用了哪个数据集?"
**中等:** "如何构建因果知识图谱?"
**困难:** "为什么图检索比纯向量检索更适合因果推理?"

### 4. 包含需要跨文档推理的问题

```python
question = """
Compare the approach in this paper with traditional RAG
methods discussed in related work. What are the key
innovations?
"""
```

## 🔧 自定义测试数据集

### 方法 1: 修改现有脚本

编辑 `generate_academic_ragas_dataset.py`:

```python
# 添加你自己的问题
custom_qa = [
    {
        "question": "你的问题",
        "ground_truth": "标准答案",
        "category": "factual"  # 或其他类型
    },
]
```

### 方法 2: 从 JSON 加载

创建 `my_questions.json`:
```json
[
  {
    "question": "What is the novelty of this approach?",
    "ground_truth": "The novelty lies in...",
    "category": "conceptual"
  }
]
```

然后在脚本中加载:
```python
with open('my_questions.json') as f:
    custom_qa = json.load(f)
```

## 📈 性能优化建议

### 如果 Context Precision 低 (< 0.6)

```yaml
# 在 config.yaml 中调整
retrieval:
  top_k: 5  # 增加检索数量
  search_type: "mmr"  # 使用 MMR 提高多样性
```

### 如果 Faithfulness 低 (< 0.7)

```python
# 在 prompt 中强调使用上下文
prompt = """
Based ONLY on the following context, answer the question.
Do not use external knowledge.

Context: {context}
Question: {question}
Answer:
"""
```

### 如果 Answer Relevancy 低 (< 0.7)

- 改进 chunking 策略(chunk_size, overlap)
- 使用更好的 embedding 模型
- 调整 prompt engineering

## 🔍 示例输出

```
============================================================
Overall RAGAS Evaluation Results
============================================================

faithfulness        : 0.8234
  → How factually accurate is the answer based on context?

answer_relevancy    : 0.7891
  → How relevant is the answer to the question?

context_precision   : 0.8456
  → Are the retrieved contexts relevant?

context_recall      : 0.7623
  → Are all relevant contexts retrieved?

============================================================
Performance Breakdown by Question Category
============================================================

FACTUAL Questions:
  faithfulness        : 0.8567
  answer_relevancy    : 0.8234
  context_precision   : 0.8901
  context_recall      : 0.8123

CONCEPTUAL Questions:
  faithfulness        : 0.7901
  answer_relevancy    : 0.7456
  context_precision   : 0.8234
  context_recall      : 0.7234

...

✓ EXCELLENT: Your RAG system performs very well on academic papers!
Overall Average Score: 0.8051
```

## 📚 参考资源

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Mistral AI Documentation](https://docs.mistral.ai/)

## 💡 常见问题

### Q: 为什么我的评分很低?
A: 可能原因:
1. 向量数据库质量不佳(检查 chunking)
2. Ground truth 质量不高
3. Retrieval 配置需要优化
4. LLM prompt 需要改进

### Q: 评估需要多长时间?
A: 对于 10-15 个问题,大约需要 5-10 分钟(取决于 API 速度)

### Q: 可以使用其他 LLM 吗?
A: 可以!修改 `evaluate_academic_ragas.py`:
```python
# 使用 OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()
```

## 📝 TODO

- [ ] 添加更多问题类型(推理、批判性分析)
- [ ] 支持多语言评估(中英文混合)
- [ ] 添加可视化报告生成
- [ ] 集成 A/B 测试框架
