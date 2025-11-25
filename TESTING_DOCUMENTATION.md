# RAG System Testing and Evaluation Framework

## Overview

This document describes the comprehensive testing and evaluation framework implemented for the Knowledge-Based Q&A System using RAG (Retrieval-Augmented Generation) architecture. The testing suite includes integration tests for multiple LLM providers and a robust RAGAS-based evaluation system for measuring RAG performance on academic papers.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Dependencies and Setup](#dependencies-and-setup)
3. [Testing Components](#testing-components)
4. [RAGAS Evaluation Framework](#ragas-evaluation-framework)
5. [Dataset Generation](#dataset-generation)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Usage Guide](#usage-guide)
8. [Results and Interpretation](#results-and-interpretation)
9. [Best Practices](#best-practices)
10. [Future Improvements](#future-improvements)

---

## System Architecture

The testing framework consists of three main components:

```
src/test/
├── test_mistralai.py                          # MistralAI integration tests
├── generate_ragas_dataset.py                  # Basic dataset generator
├── generate_comprehensive_ragas_dataset.py     # Comprehensive multi-category dataset
├── generate_academic_ragas_dataset.py         # Academic paper-focused dataset
├── evaluate_academic_ragas.py                 # RAGAS evaluation runner
└── test_data/                                 # Generated datasets and results
    ├── ragas_dataset.json
    ├── comprehensive_ragas_dataset.json
    ├── academic_ragas_dataset.json
    └── academic_ragas_results.json
```

### Key Technologies

- **LLM Provider**: MistralAI (mistral-small-latest)
- **Embeddings**: MistralAI Embeddings (mistral-embed)
- **Evaluation Framework**: RAGAS (Retrieval-Augmented Generation Assessment)
- **Vector Database**: ChromaDB
- **Orchestration**: LangChain

---

## Dependencies and Setup

### Required Dependencies

The following dependencies have been added to `requirements.txt`:

```
# Testing and Evaluation
langchain-mistralai>=0.1.0    # MistralAI integration for LangChain
ragas>=0.2.0                   # RAG evaluation framework
datasets>=2.14.0               # Dataset handling for RAGAS
```

### Configuration Updates

The `config.yaml` has been extended to support:

1. **MistralAI as an LLM Provider**:
```yaml
llm:
  provider: "mistralai"
  mistralai_model: "mistral-small-latest"
```

2. **MistralAI Embeddings**:
```yaml
embeddings:
  provider: "mistralai"
  mistralai_model: "mistral-embed"
```

3. **Testing Configuration**:
```yaml
testing:
  ragas:
    enabled: true
    output_directory: "./src/test/test_data"
    metrics: ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
  dataset_generation:
    default_k: 4
    categories: ["factual", "conceptual", "methodology", "comparative", "application"]
```

### Environment Setup

Create a `.env` file with your MistralAI API key:

```bash
MISTRAL_API_KEY=your_api_key_here
```

---

## Testing Components

### 1. MistralAI Integration Test (`test_mistralai.py`)

**Purpose**: Verify MistralAI API connectivity and basic functionality.

**Features**:
- Tests chat completion with MistralAI
- Tests embedding generation
- Validates API key configuration

**Usage**:
```bash
python src/test/test_mistralai.py
```

### 2. Basic Dataset Generator (`generate_ragas_dataset.py`)

**Purpose**: Generate simple RAGAS test samples for initial validation.

**Features**:
- Retrieves relevant documents from vector store
- Generates answers using MistralAI
- Creates RAGAS-compatible dataset format

**Dataset Structure**:
```json
{
  "question": "How to use GraphRAG for causal reasoning?",
  "answer": "Generated answer from the RAG system...",
  "contexts": ["Retrieved document 1...", "Retrieved document 2..."],
  "ground_truth": "Reference answer for evaluation..."
}
```

### 3. Comprehensive Dataset Generator (`generate_comprehensive_ragas_dataset.py`)

**Purpose**: Generate extensive test datasets with multiple question types and difficulty levels.

**Features**:
- **8 Question Categories**:
  1. Single-document Factual
  2. Single-document Conceptual
  3. Cross-document Comparative
  4. Cross-document Synthesis
  5. Methodological Deep Dive
  6. Application & Extension
  7. Critical Analysis
  8. Edge Cases

- **3 Difficulty Levels**: Easy, Medium, Hard
- **Adaptive Retrieval**: Adjusts k (number of retrieved documents) based on difficulty
- **Rich Metadata**: Includes category, difficulty, and context count

**Statistics Tracking**:
- Total samples
- Breakdown by category and difficulty
- Average answer, ground truth, and context lengths

### 4. Academic-Focused Dataset Generator (`generate_academic_ragas_dataset.py`)

**Purpose**: Generate specialized datasets for evaluating RAG performance on academic papers.

**Features**:
- **5 Question Types**:
  1. **Factual**: Direct information retrieval from papers
  2. **Conceptual**: Understanding of key concepts and theories
  3. **Methodology**: Technical implementation details
  4. **Comparative**: Comparison between different approaches
  5. **Application**: Real-world use cases and extensions

- **RAGAS 0.2+ Compatible**: Uses standardized field names
  - `user_input`: The question
  - `response`: Generated answer
  - `retrieved_contexts`: List of retrieved document chunks
  - `reference`: Ground truth answer

**Sample Questions**:
- "What is the main contribution of the GraphRAG-Causal framework?"
- "How does graph-based retrieval enhance causal reasoning?"
- "What are the advantages of GraphRAG-Causal over standard RAG?"

---

## RAGAS Evaluation Framework

### What is RAGAS?

RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG systems using LLM-based metrics that assess both retrieval quality and generation quality.

### Why RAGAS?

Traditional metrics (BLEU, ROUGE) only measure surface-level similarity and fail to capture:
- Semantic relevance
- Factual correctness
- Context utilization
- Answer completeness

RAGAS uses LLMs to evaluate these aspects more accurately.

---

## Evaluation Metrics

### 1. Faithfulness (0.0 - 1.0)

**Definition**: Measures whether the generated answer is factually consistent with the retrieved contexts.

**Evaluation**:
- Checks if claims in the answer can be verified from the context
- Detects hallucinations and extrapolations

**Interpretation**:
- **High (>0.8)**: Answer strictly follows retrieved information
- **Medium (0.6-0.8)**: Some extrapolation present
- **Low (<0.6)**: Significant hallucination or fabrication

**Improvement Strategies**:
- Adjust prompts to emphasize using only provided context
- Lower LLM temperature for more deterministic outputs
- Use retrieval-aware prompting techniques

### 2. Answer Relevancy (0.0 - 1.0)

**Definition**: Measures how relevant the generated answer is to the asked question.

**Evaluation**:
- Compares semantic similarity between question and answer
- Checks if the answer addresses the question directly

**Interpretation**:
- **High (>0.8)**: Answer directly addresses the question
- **Medium (0.6-0.8)**: Partially relevant answer
- **Low (<0.6)**: Off-topic or tangential answer

**Improvement Strategies**:
- Improve prompt engineering
- Ensure question-answer alignment in training
- Use question decomposition for complex queries

### 3. Context Precision (0.0 - 1.0)

**Definition**: Measures how many of the retrieved contexts are actually relevant to the question.

**Evaluation**:
- Assesses the precision of the retrieval system
- Checks if irrelevant documents were retrieved

**Interpretation**:
- **High (>0.8)**: Most retrieved contexts are relevant
- **Medium (0.6-0.8)**: Some irrelevant contexts retrieved
- **Low (<0.6)**: Many irrelevant contexts retrieved

**Improvement Strategies**:
- Optimize chunking strategy (size, overlap)
- Use better embedding models
- Implement re-ranking mechanisms
- Apply hybrid search (semantic + keyword)

### 4. Context Recall (0.0 - 1.0)

**Definition**: Measures whether all relevant information needed to answer the question was retrieved.

**Evaluation**:
- Compares retrieved contexts against ground truth
- Checks if any important information was missed

**Interpretation**:
- **High (>0.8)**: All necessary information retrieved
- **Medium (0.6-0.8)**: Some relevant info missing
- **Low (<0.6)**: Significant information gaps

**Improvement Strategies**:
- Increase top_k (retrieve more documents)
- Use Maximum Marginal Relevance (MMR) for diversity
- Implement query expansion techniques
- Add metadata filtering

---

## Usage Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Vector Database

Ensure your vector database is populated with documents:

```bash
# Check if ChromaDB exists
ls -lh chroma_db/

# If not, run document ingestion
python ingest_documents.py
```

### Step 3: Generate Test Dataset

Choose the appropriate dataset generator:

**For Quick Testing**:
```bash
python src/test/generate_ragas_dataset.py
```

**For Comprehensive Evaluation**:
```bash
python src/test/generate_comprehensive_ragas_dataset.py
```

**For Academic Papers** (Recommended):
```bash
python src/test/generate_academic_ragas_dataset.py
```

**Output**: A JSON file in `src/test/test_data/` with the generated dataset.

### Step 4: Run RAGAS Evaluation

```bash
python src/test/evaluate_academic_ragas.py
```

**Output**:
- Console output with detailed metrics
- `academic_ragas_results.json` with numerical scores
- Performance breakdown by question category
- Recommendations for improvement

---

## Results and Interpretation

### Sample Output

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
  faithfulness        : 0.8567 (n=3)
  answer_relevancy    : 0.8234 (n=3)
  context_precision   : 0.8901 (n=3)
  context_recall      : 0.8123 (n=3)

CONCEPTUAL Questions:
  faithfulness        : 0.7901 (n=3)
  answer_relevancy    : 0.7456 (n=3)
  context_precision   : 0.8234 (n=3)
  context_recall      : 0.7234 (n=3)

...

✓ EXCELLENT: Your RAG system performs very well on academic papers!
Overall Average Score: 0.8051
```

### Performance Bands

| Score Range | Grade | Description |
|------------|-------|-------------|
| 0.8 - 1.0 | Excellent | RAG system performs exceptionally well |
| 0.6 - 0.8 | Good | Solid performance with room for improvement |
| 0.4 - 0.6 | Moderate | Optimization needed for retrieval or generation |
| < 0.4 | Needs Improvement | Significant changes required |

### Interpreting Category Performance

**Strong Factual Performance (>0.8)**:
- Your RAG excels at retrieving and presenting facts
- Document chunking is effective
- Embedding model captures semantic similarity well

**Weak Conceptual Performance (<0.6)**:
- Struggles with abstract reasoning
- May need larger context windows
- Consider using more sophisticated prompting

**Poor Application Performance (<0.5)**:
- Difficulty generalizing knowledge
- May need few-shot examples
- Consider fine-tuning for specific domains

---

## Best Practices

### 1. Designing Quality Ground Truth

**Good Ground Truth**:
```python
ground_truth = """
GraphRAG-Causal combines graph-based retrieval with LLMs
to enhance causal reasoning through three key innovations:
(1) Structuring news as causal knowledge graphs
(2) Using hybrid semantic + graph queries for retrieval
(3) Employing XML-based prompting for few-shot learning.
"""
```

**Poor Ground Truth**:
```python
ground_truth = "It's a framework."  # Too vague
```

### 2. Question Design Principles

- **Specific**: Target particular aspects of the documents
- **Varied**: Cover different difficulty levels and types
- **Answerable**: Ensure information exists in the corpus
- **Realistic**: Mimic actual user queries

### 3. Dataset Composition

Recommended distribution:
- 30% Factual (easy-medium)
- 25% Conceptual (medium)
- 20% Comparative (medium-hard)
- 15% Application (hard)
- 10% Edge Cases (hard)

### 4. Iterative Improvement Workflow

1. **Baseline Evaluation**: Run RAGAS on initial dataset
2. **Identify Weaknesses**: Analyze per-category and per-metric scores
3. **Targeted Optimization**: Focus on lowest-scoring areas
4. **Re-evaluate**: Measure improvement
5. **Repeat**: Continue until satisfactory performance

### 5. Configuration Tuning

**For Better Precision** (reduce irrelevant contexts):
```yaml
retrieval:
  top_k: 3
  score_threshold: 0.8
  search_type: "similarity"
```

**For Better Recall** (capture more information):
```yaml
retrieval:
  top_k: 6
  score_threshold: 0.6
  search_type: "mmr"
```

---

## Technical Implementation Details

### RAGDatasetGenerator Class

**Key Methods**:

1. `__init__()`: Initializes vector store and MistralAI client
2. `generate_sample(question, ground_truth, category)`: Generates a single test sample
   - Retrieves relevant contexts from vector DB
   - Generates answer using MistralAI
   - Formats in RAGAS-compatible structure
3. `save_dataset(dataset, path)`: Saves dataset to JSON
4. `print_statistics(dataset)`: Displays dataset statistics

### Evaluation Pipeline

**Workflow**:
```python
# 1. Load dataset
dataset = load_dataset("academic_ragas_dataset.json")

# 2. Initialize models
llm = ChatMistralAI(model="mistral-small-latest")
embeddings = MistralAIEmbeddings(model="mistral-embed")

# 3. Run evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings
)

# 4. Analyze results
print_overall_scores(result)
print_category_breakdown(result, dataset_json)
```

### RAGAS Field Mapping (v0.2+)

| RAGAS Field | Description |
|------------|-------------|
| `user_input` | The question asked by the user |
| `response` | The answer generated by the RAG system |
| `retrieved_contexts` | List of document chunks retrieved |
| `reference` | Ground truth answer for comparison |

---

## Troubleshooting

### Common Issues

**Issue 1: Low Faithfulness Scores**

**Symptoms**: Faithfulness < 0.6

**Diagnosis**:
- Model is hallucinating or adding external knowledge
- Retrieved contexts don't contain sufficient information

**Solutions**:
```python
# Adjust prompt to emphasize context-only answers
prompt = """
Answer ONLY based on the provided context.
Do not use external knowledge.
If the context doesn't contain the answer, say "I don't have enough information."

Context: {context}
Question: {question}
Answer:
"""
```

**Issue 2: Low Context Recall**

**Symptoms**: Context recall < 0.6

**Diagnosis**:
- Not retrieving all relevant documents
- Chunk size too small/large
- Ground truth expects information not in corpus

**Solutions**:
- Increase `top_k` in config.yaml
- Adjust `chunk_size` and `chunk_overlap`
- Verify ground truth is answerable from documents

**Issue 3: RAGAS Evaluation Fails**

**Symptoms**: Error during evaluation

**Possible Causes**:
- Field name mismatch (using old RAGAS format)
- API key issues
- Empty contexts or responses

**Solutions**:
```bash
# Check RAGAS version
pip show ragas

# Validate dataset format
python -c "
import json
with open('academic_ragas_dataset.json') as f:
    data = json.load(f)
    print('Fields:', list(data[0].keys()))
    # Should see: user_input, response, retrieved_contexts, reference
"
```

---

## Future Improvements

### Planned Enhancements

1. **Multi-lingual Evaluation**
   - Support for Chinese-English mixed datasets
   - Cross-lingual retrieval testing

2. **Advanced Metrics**
   - Citation accuracy
   - Answer completeness
   - Multi-hop reasoning capability

3. **Automated Hyperparameter Tuning**
   - Grid search for optimal chunk size
   - Auto-tuning of top_k and temperature

4. **Visualization Dashboard**
   - Interactive performance charts
   - Per-question drill-down analysis
   - Temporal performance tracking

5. **A/B Testing Framework**
   - Compare different embedding models
   - Test multiple LLM providers
   - Evaluate prompt variations

6. **Real-time Monitoring**
   - Production RAG performance tracking
   - Alert on quality degradation
   - Automated retraining triggers

---

## Conclusion

This testing framework provides a comprehensive solution for evaluating RAG systems on academic papers. By combining:

- **MistralAI Integration**: State-of-the-art LLM and embeddings
- **RAGAS Metrics**: LLM-based evaluation of retrieval and generation quality
- **Multi-category Datasets**: Diverse question types covering different aspects
- **Detailed Analysis**: Per-category breakdowns and improvement recommendations

The framework enables systematic improvement of RAG systems through data-driven insights.

### Key Achievements

✅ Integrated MistralAI for advanced language understanding
✅ Implemented RAGAS-based evaluation pipeline
✅ Created comprehensive test datasets with 5+ question categories
✅ Provided actionable metrics and improvement recommendations
✅ Established best practices for RAG system evaluation

### References

1. [RAGAS Documentation](https://docs.ragas.io/)
2. [MistralAI Documentation](https://docs.mistral.ai/)
3. [LangChain Documentation](https://python.langchain.com/)
4. Lewis, P., et al. (2020). "Retrieval-augmented generation for knowledge-intensive NLP tasks."

---

**Last Updated**: 2025-11-25
**Author**: Course Project - Knowledge-Based Q&A System
**Version**: 1.0
