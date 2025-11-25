"""
Generate comprehensive RAGAS dataset with cross-document reasoning
生成包含跨文档推理的综合 RAGAS 数据集
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

from src.vector_store import VectorStoreManager
from langchain_mistralai import ChatMistralAI


class ComprehensiveRAGASGenerator:
    """Generate comprehensive RAGAS dataset with multiple question types"""

    def __init__(self):
        print("Initializing Comprehensive RAGAS Generator...")
        config_path = project_root / "config.yaml"
        self.vector_store_manager = VectorStoreManager(config_path=str(config_path))
        self.agent = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-small-latest",
            temperature=0.1  # 低温度保证一致性
        )

    def generate_sample(self, question, ground_truth, category, difficulty="medium"):
        """Generate a single sample with metadata"""
        print(f"\n{'='*60}")
        print(f"Category: {category} | Difficulty: {difficulty}")
        print(f"Question: {question[:80]}...")

        # 根据难度调整检索数量
        k_values = {"easy": 2, "medium": 4, "hard": 6}
        k = k_values.get(difficulty, 4)

        # 检索相关文档
        retrieved_docs = self.vector_store_manager.similarity_search(question, k=k)
        contexts = [doc.page_content for doc in retrieved_docs]
        print(f"Retrieved: {len(contexts)} contexts")

        # 生成答案
        response = self.agent.invoke(question)
        answer = response.content
        print(f"Generated: {len(answer)} chars")

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "category": category,
            "difficulty": difficulty,
            "num_contexts": len(contexts),
        }

    def generate_all_questions(self):
        """Generate all test questions"""
        questions = []

        # ============================================================
        # 1. 单文档事实性问题 (Single-doc Factual)
        # ============================================================
        questions.extend([
            {
                "question": "What is the main contribution of the GraphRAG-Causal framework?",
                "ground_truth": "GraphRAG-Causal combines graph-based retrieval with LLMs to enhance causal reasoning in news analysis by transforming headlines into structured causal knowledge graphs with a three-stage pipeline.",
                "category": "single_doc_factual",
                "difficulty": "easy"
            },
            {
                "question": "What datasets are used to evaluate GraphRAG-Causal?",
                "ground_truth": "The framework is evaluated on causal news datasets with annotated causal relationships in headlines.",
                "category": "single_doc_factual",
                "difficulty": "easy"
            },
            {
                "question": "What are the three stages in the GraphRAG-Causal pipeline?",
                "ground_truth": "The three stages are: (1) Data Preparation - converting sentences into causal graphs, (2) Graph Retrieval - using Neo4j with hybrid queries, (3) LLM Inference - few-shot learning with XML prompting.",
                "category": "single_doc_factual",
                "difficulty": "medium"
            },
        ])

        # ============================================================
        # 2. 单文档概念理解 (Single-doc Conceptual)
        # ============================================================
        questions.extend([
            {
                "question": "Why is graph-based retrieval important for causal reasoning?",
                "ground_truth": "Graph-based retrieval captures structural causal relationships between entities, enabling multi-hop reasoning and understanding of complex causal chains that vector-only methods might miss.",
                "category": "single_doc_conceptual",
                "difficulty": "medium"
            },
            {
                "question": "How does XML-based prompting improve causal classification?",
                "ground_truth": "XML-based prompting provides structured contextual cues that help the LLM better parse and understand the retrieved causal graph information, leading to more accurate classification.",
                "category": "single_doc_conceptual",
                "difficulty": "medium"
            },
            {
                "question": "What is the role of few-shot learning in GraphRAG-Causal?",
                "ground_truth": "Few-shot learning provides the LLM with example causal patterns from the retrieved graphs, enabling better generalization to new causal relationships without extensive fine-tuning.",
                "category": "single_doc_conceptual",
                "difficulty": "hard"
            },
        ])

        # ============================================================
        # 3. 跨文档对比分析 (Cross-doc Comparative)
        # ============================================================
        questions.extend([
            {
                "question": "How does GraphRAG-Causal differ from standard RAG approaches?",
                "ground_truth": "GraphRAG-Causal extends standard RAG by incorporating structured knowledge graphs for retrieval, combining semantic embeddings with graph topology, whereas standard RAG relies solely on vector similarity.",
                "category": "cross_doc_comparative",
                "difficulty": "medium"
            },
            {
                "question": "Compare graph-based retrieval with vector-only retrieval for causal reasoning tasks.",
                "ground_truth": "Graph-based retrieval captures explicit causal relationships and enables multi-hop reasoning through graph structure, while vector-only retrieval relies on semantic similarity which may miss structural causal patterns. Graphs provide better interpretability and precision for causal tasks.",
                "category": "cross_doc_comparative",
                "difficulty": "hard"
            },
            {
                "question": "What are the trade-offs between using knowledge graphs versus pure neural approaches for RAG?",
                "ground_truth": "Knowledge graphs provide explicit structure, interpretability, and precise relationship modeling but require manual construction and maintenance. Pure neural approaches are more scalable and flexible but may lack interpretability and struggle with complex logical reasoning.",
                "category": "cross_doc_comparative",
                "difficulty": "hard"
            },
        ])

        # ============================================================
        # 4. 跨文档综合推理 (Cross-doc Synthesis)
        # ============================================================
        questions.extend([
            {
                "question": "How can RAG systems be improved using techniques from both retrieval and generation research?",
                "ground_truth": "RAG systems can be improved by combining advanced retrieval techniques (hybrid search, re-ranking, query expansion) with generation improvements (instruction tuning, chain-of-thought, constrained generation). Integrating structured knowledge and iterative refinement further enhances performance.",
                "category": "cross_doc_synthesis",
                "difficulty": "hard"
            },
            {
                "question": "What are the common challenges in deploying RAG systems in production, and how can they be addressed?",
                "ground_truth": "Common challenges include: latency (addressed by caching and efficient indexing), retrieval quality (improved by hybrid search and re-ranking), hallucination (mitigated by faithfulness metrics and citation), and maintenance (solved by automated updates and monitoring). Cost and scalability require infrastructure optimization.",
                "category": "cross_doc_synthesis",
                "difficulty": "hard"
            },
        ])

        # ============================================================
        # 5. 方法论深度理解 (Methodological Deep Dive)
        # ============================================================
        questions.extend([
            {
                "question": "Explain the hybrid query mechanism in GraphRAG-Causal in detail.",
                "ground_truth": "The hybrid query mechanism combines semantic embeddings (for similarity search) with graph-based structural queries (for relationship traversal). It first retrieves semantically similar nodes, then expands the search through graph edges to find connected causal patterns, leveraging both semantic and structural information.",
                "category": "methodological",
                "difficulty": "hard"
            },
            {
                "question": "How are causal knowledge graphs constructed from unstructured text?",
                "ground_truth": "Construction involves: (1) Entity extraction using NER, (2) Relationship extraction to identify causal links (cause, effect, trigger), (3) Graph structuring with nodes as entities/events and edges as relationships, (4) Annotation and validation, (5) Integration into a graph database like Neo4j.",
                "category": "methodological",
                "difficulty": "hard"
            },
        ])

        # ============================================================
        # 6. 应用场景与扩展 (Application & Extension)
        # ============================================================
        questions.extend([
            {
                "question": "How can GraphRAG-Causal be adapted for financial news analysis?",
                "ground_truth": "Adapt by: (1) Building a financial causal knowledge graph with entities like companies, markets, policies, (2) Training on financial news with causal annotations, (3) Incorporating temporal information for time-series causal chains, (4) Adding domain-specific metrics and evaluation criteria.",
                "category": "application",
                "difficulty": "medium"
            },
            {
                "question": "What are potential limitations of GraphRAG-Causal in real-world deployment?",
                "ground_truth": "Limitations include: (1) Graph maintenance overhead for dynamic domains, (2) Computational cost of graph queries, (3) Difficulty handling ambiguous or evolving causal relationships, (4) Dependence on quality of initial graph construction, (5) Latency in real-time applications.",
                "category": "application",
                "difficulty": "medium"
            },
            {
                "question": "How could GraphRAG-Causal be extended to multi-lingual causal reasoning?",
                "ground_truth": "Extension requires: (1) Multi-lingual embeddings for cross-lingual retrieval, (2) Translated or aligned knowledge graphs, (3) Language-agnostic graph representations, (4) Multi-lingual causal pattern templates, (5) Evaluation on multi-lingual causal datasets.",
                "category": "application",
                "difficulty": "hard"
            },
        ])

        # ============================================================
        # 7. 批判性分析 (Critical Analysis)
        # ============================================================
        questions.extend([
            {
                "question": "What are the assumptions and potential biases in GraphRAG-Causal?",
                "ground_truth": "Assumptions include: (1) Causal relationships can be explicitly modeled in graphs, (2) Historical causal patterns generalize to new cases, (3) LLMs can accurately interpret graph structures. Potential biases include: annotation bias in training data, over-reliance on structural cues, and dataset-specific patterns that may not transfer.",
                "category": "critical_analysis",
                "difficulty": "hard"
            },
            {
                "question": "Under what circumstances might standard RAG outperform GraphRAG-Causal?",
                "ground_truth": "Standard RAG may outperform when: (1) Causal relationships are not the primary focus, (2) Data is highly unstructured without clear relationships, (3) Graph construction is infeasible or costly, (4) Low-latency requirements make graph queries impractical, (5) The domain lacks sufficient causal patterns for graph-based benefits.",
                "category": "critical_analysis",
                "difficulty": "hard"
            },
        ])

        # ============================================================
        # 8. 边界测试 (Edge Cases)
        # ============================================================
        questions.extend([
            {
                "question": "How does GraphRAG-Causal handle ambiguous or contradictory causal relationships?",
                "ground_truth": "The framework can handle ambiguity through: (1) Probabilistic graph edges with confidence scores, (2) Multi-path retrieval to surface competing explanations, (3) LLM-based disambiguation using context, (4) Explicit representation of uncertainty in causal claims.",
                "category": "edge_cases",
                "difficulty": "hard"
            },
            {
                "question": "What happens when the knowledge graph has missing or incomplete causal information?",
                "ground_truth": "With incomplete graphs: (1) Retrieval may return sparse results, (2) The system can fall back to semantic similarity, (3) LLM may rely more on parametric knowledge (risking hallucination), (4) Confidence scores should reflect uncertainty. Solutions include graph completion techniques and hybrid retrieval strategies.",
                "category": "edge_cases",
                "difficulty": "hard"
            },
        ])

        return questions

    def generate_dataset(self):
        """Generate the complete dataset"""
        questions = self.generate_all_questions()

        print(f"\n{'='*60}")
        print(f"Generating {len(questions)} RAGAS samples...")
        print(f"{'='*60}")

        dataset = []
        for i, q in enumerate(questions, 1):
            print(f"\nProgress: {i}/{len(questions)}")
            sample = self.generate_sample(
                q["question"],
                q["ground_truth"],
                q["category"],
                q["difficulty"]
            )
            dataset.append(sample)

        return dataset

    def save_dataset(self, dataset, output_path):
        """Save dataset with pretty formatting"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Dataset saved to: {output_file}")

    def print_statistics(self, dataset):
        """Print comprehensive statistics"""
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}\n")

        # Total
        print(f"Total samples: {len(dataset)}")

        # By category
        categories = {}
        for sample in dataset:
            cat = sample['category']
            categories[cat] = categories.get(cat, 0) + 1

        print("\nBy Category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat:25s}: {count:2d}")

        # By difficulty
        difficulties = {}
        for sample in dataset:
            diff = sample['difficulty']
            difficulties[diff] = difficulties.get(diff, 0) + 1

        print("\nBy Difficulty:")
        for diff in ["easy", "medium", "hard"]:
            count = difficulties.get(diff, 0)
            print(f"  {diff:10s}: {count:2d}")

        # Averages
        avg_answer = sum(len(s['answer']) for s in dataset) / len(dataset)
        avg_contexts = sum(s['num_contexts'] for s in dataset) / len(dataset)
        avg_ground_truth = sum(len(s['ground_truth']) for s in dataset) / len(dataset)

        print(f"\nAverage Lengths:")
        print(f"  Answer:       {avg_answer:6.0f} chars")
        print(f"  Ground Truth: {avg_ground_truth:6.0f} chars")
        print(f"  Contexts:     {avg_contexts:6.1f} per question")

        print(f"\n{'='*60}")


def main():
    """Main function"""
    generator = ComprehensiveRAGASGenerator()

    # Generate dataset
    dataset = generator.generate_dataset()

    # Save
    output_path = project_root / "src/test/test_data/comprehensive_ragas_dataset.json"
    generator.save_dataset(dataset, output_path)

    # Statistics
    generator.print_statistics(dataset)

    # Preview
    print(f"\n{'='*60}")
    print("Sample Preview")
    print(f"{'='*60}")
    sample = dataset[0]
    print(f"\nCategory:   {sample['category']}")
    print(f"Difficulty: {sample['difficulty']}")
    print(f"\nQuestion:\n{sample['question']}")
    print(f"\nAnswer (first 250 chars):\n{sample['answer'][:250]}...")
    print(f"\nGround Truth:\n{sample['ground_truth']}")
    print(f"\nContexts: {sample['num_contexts']}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
