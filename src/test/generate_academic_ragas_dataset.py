"""
Generate RAGAS dataset specifically designed for AI research papers
针对 AI 论文的 RAGAS 评估数据集生成
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


class AcademicRAGASDatasetGenerator:
    """Generate RAGAS dataset for academic papers"""

    def __init__(self):
        print("Initializing Academic RAGAS Dataset Generator...")
        config_path = project_root / "config.yaml"
        self.vector_store_manager = VectorStoreManager(config_path=str(config_path))
        self.agent = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-small-latest"
        )

    def generate_sample(self, question, ground_truth, category):
        """Generate a single RAGAS sample"""
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"Question: {question}")

        # Retrieve relevant documents
        retrieved_docs = self.vector_store_manager.similarity_search(question, k=4)
        contexts = [doc.page_content for doc in retrieved_docs]
        print(f"Retrieved: {len(contexts)} contexts")

        # Generate answer using the agent
        response = self.agent.invoke(question)
        answer = response.content
        print(f"Generated answer length: {len(answer)} chars")

        # 使用 RAGAS 0.2+ 标准字段名，同时保留 category 用于分类统计
        return {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth,
            "category": category,  # 保留类别用于分类评估
        }

    def generate_comprehensive_dataset(self):
        """Generate a comprehensive dataset covering multiple question types"""

        # 1. 事实性问题 (Factual)
        factual_qa = [
            {
                "question": "What is the main contribution of the GraphRAG-Causal framework?",
                "ground_truth": "GraphRAG-Causal combines graph-based retrieval with Large Language Models to enhance causal reasoning and annotation in news analysis, addressing the challenges of identifying complex, implicit causal links by transforming news headlines into structured causal knowledge graphs.",
                "category": "factual"
            },
            {
                "question": "What datasets are used to evaluate the GraphRAG-Causal framework?",
                "ground_truth": "The framework is evaluated on causal news datasets, specifically using news headlines annotated with causal relationships.",
                "category": "factual"
            },
            {
                "question": "What are the three main stages in the GraphRAG-Causal pipeline?",
                "ground_truth": "The three stages are: (1) Data Preparation - annotating sentences and converting them into causal graphs, (2) Graph Retrieval - using Neo4j database with hybrid query mechanism, and (3) LLM Inference - using retrieved graphs for causal classification with XML-based prompting.",
                "category": "factual"
            },
        ]

        # 2. 概念理解问题 (Conceptual)
        conceptual_qa = [
            {
                "question": "How does graph-based retrieval enhance causal reasoning compared to traditional RAG?",
                "ground_truth": "Graph-based retrieval captures structural relationships and causal connections between entities, enabling the model to understand complex, multi-hop causal chains that traditional vector-only RAG might miss. It combines semantic embeddings with graph structure for more precise retrieval.",
                "category": "conceptual"
            },
            {
                "question": "Why is XML-based prompting used in the GraphRAG-Causal framework?",
                "ground_truth": "XML-based prompting provides proper contextual cues and structured formatting that helps the LLM better understand and process the retrieved causal graph information for accurate causal classification.",
                "category": "conceptual"
            },
            {
                "question": "What role does the knowledge graph play in causal reasoning?",
                "ground_truth": "The knowledge graph stores structured causal relationships between entities, enabling the system to retrieve relevant causal patterns and precedents, which helps in identifying implicit causal links in new text.",
                "category": "conceptual"
            },
        ]

        # 3. 方法论问题 (Methodology)
        methodology_qa = [
            {
                "question": "How is the hybrid query mechanism implemented in GraphRAG-Causal?",
                "ground_truth": "The hybrid query mechanism combines semantic embeddings (for similarity search) with graph-based structural queries, leveraging both vector representations and graph topology to retrieve relevant causal patterns from the Neo4j database.",
                "category": "methodology"
            },
            {
                "question": "What is the process for constructing causal knowledge graphs from news headlines?",
                "ground_truth": "News headlines are annotated to identify causal relationships, then converted into graph structures where nodes represent entities or events, and edges represent causal links (cause, effect, trigger relationships).",
                "category": "methodology"
            },
        ]

        # 4. 对比分析问题 (Comparative)
        comparative_qa = [
            {
                "question": "What are the advantages of GraphRAG-Causal over standard RAG approaches?",
                "ground_truth": "GraphRAG-Causal provides better causal reasoning by leveraging structured graph relationships, enabling multi-hop reasoning and capturing implicit causal connections that vector-only approaches might miss. It also offers more interpretable retrieval through graph structures.",
                "category": "comparative"
            },
            {
                "question": "How does few-shot learning with graphs differ from traditional few-shot prompting?",
                "ground_truth": "Few-shot learning with graphs provides structured examples that include not just text but also relational information, giving the model richer context about causal patterns and their relationships, leading to better generalization.",
                "category": "comparative"
            },
        ]

        # 5. 应用场景问题 (Application)
        application_qa = [
            {
                "question": "How can GraphRAG-Causal be applied to financial news analysis?",
                "ground_truth": "GraphRAG-Causal can analyze financial news by identifying causal relationships between market events, corporate actions, and economic outcomes. It can trace causal chains like 'policy change → market reaction → stock price movement' to provide evidence-based financial analysis.",
                "category": "application"
            },
            {
                "question": "What are potential challenges in deploying GraphRAG-Causal for real-time news analysis?",
                "ground_truth": "Challenges include: maintaining and updating the knowledge graph with real-time data, computational overhead of graph queries, handling ambiguous or evolving causal relationships, and ensuring low latency for real-time applications.",
                "category": "application"
            },
        ]

        # Combine all questions
        all_qa = (
            factual_qa +
            conceptual_qa +
            methodology_qa +
            comparative_qa +
            application_qa
        )

        # Generate samples
        dataset = []
        total = len(all_qa)

        print(f"\n{'='*60}")
        print(f"Generating {total} RAGAS samples...")
        print(f"{'='*60}")

        for i, qa in enumerate(all_qa, 1):
            print(f"\nProgress: {i}/{total}")
            sample = self.generate_sample(
                qa["question"],
                qa["ground_truth"],
                qa["category"]
            )
            dataset.append(sample)

        return dataset

    def save_dataset(self, dataset, output_path):
        """Save dataset to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Dataset saved to: {output_file}")

    def print_statistics(self, dataset):
        """Print dataset statistics"""
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print(f"{'='*60}")

        # Count by category
        categories = {}
        for sample in dataset:
            cat = sample['category']
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\nTotal samples: {len(dataset)}")
        print("\nBreakdown by category:")
        for cat, count in categories.items():
            print(f"  {cat:15s}: {count:2d} samples")

        # Average lengths - 使用正确的 RAGAS 字段名
        avg_answer_len = sum(len(s['response']) for s in dataset) / len(dataset)
        avg_context_len = sum(len(s['retrieved_contexts']) for s in dataset) / len(dataset)
        avg_ground_truth_len = sum(len(s['reference']) for s in dataset) / len(dataset)

        print(f"\nAverage lengths:")
        print(f"  Response:      {avg_answer_len:.0f} chars")
        print(f"  Reference:     {avg_ground_truth_len:.0f} chars")
        print(f"  Contexts:      {avg_context_len:.1f} per question")

        print(f"{'='*60}\n")


def main():
    """Main function"""
    generator = AcademicRAGASDatasetGenerator()

    # Generate comprehensive dataset
    dataset = generator.generate_comprehensive_dataset()

    # Save dataset
    output_path = project_root / "src/test/test_data/academic_ragas_dataset.json"
    generator.save_dataset(dataset, output_path)

    # Print statistics
    generator.print_statistics(dataset)

    # Preview first sample - 使用正确的 RAGAS 字段名
    print(f"{'='*60}")
    print("Sample Preview (First Entry):")
    print(f"{'='*60}")
    sample = dataset[0]
    print(f"\nCategory: {sample['category']}")
    print(f"\nUser Input:\n{sample['user_input']}")
    print(f"\nResponse (first 300 chars):\n{sample['response'][:300]}...")
    print(f"\nReference:\n{sample['reference']}")
    print(f"\nNumber of contexts: {len(sample['retrieved_contexts'])}")
    if sample['retrieved_contexts']:
        print(f"\nFirst context (first 200 chars):\n{sample['retrieved_contexts'][0][:200]}...")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()