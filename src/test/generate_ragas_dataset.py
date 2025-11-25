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
class RAGDatasetGenerator:
    """generate a RAG dataset using MistralAI embeddings and a vector store."""

    def __init__(self):
        print("Initializing RAG Dataset Generator...")
        print("Initializing Vector Store")
        # 传入正确的 config 路径
        config_path = project_root / "config.yaml"
        self.vector_store_manager = VectorStoreManager(config_path=str(config_path))
        print("Initializing Agent")
        self.agent = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model_name="mistral-small-latest")

    def generate_dataset(self, question, ground_truth=None):
        """Generate a RAG dataset for a single question."""
        print(f"\nTracking question: {question}")

        print("Retrieving relevant documents from vector store")
        retrieved_docs = self.vector_store_manager.similarity_search(question, k=3)
        
        contexts = [doc.page_content for doc in retrieved_docs]

        print(f" - retrieved {len(retrieved_docs)} documents")

        print("Generating answer using agent")

        response = self.agent.invoke(question)
        answer = response.content

        print(f" - generated answer: {answer}")

        sample = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth or ""
        }

        return sample
    
    def save_dataset(self, dataset, output_path):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
            print(f"Dataset saved to {output_file}")
    
    def load_dataset(self, input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        print(f"Loaded dataset from {input_path} with {len(dataset)} samples")
        return dataset
    
def main():
    test_questions = [
        "How to use GraphRAG to enhance causal reasoning in news analysis?",
    ]

    ground_truths = [
        "Use GraphRAG-Causal framework to integrate knowledge graphs with RAG models for improved causal reasoning in news analysis.",
    ]
    generator = RAGDatasetGenerator()

    sample = generator.generate_dataset(test_questions[0], ground_truths[0])

    # 将单个样本包装成列表
    dataset = [sample]

    output_path = project_root / "src/test/test_data/ragas_dataset.json"

    generator.save_dataset(dataset, output_path)

    # 打印样本预览
    print("\n" + "="*60)
    print("Dataset Preview:")
    print("="*60)
    print(f"\nQuestion: {sample['question']}")
    print(f"\nAnswer: {sample['answer'][:200]}...")
    print(f"\nNumber of contexts: {len(sample['contexts'])}")
    if sample['contexts']:
        print(f"First context: {sample['contexts'][0][:150]}...")
    print(f"\nGround truth: {sample['ground_truth'][:200]}...")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()