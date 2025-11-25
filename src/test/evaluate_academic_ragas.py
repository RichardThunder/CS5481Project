"""
Evaluate RAG system on academic papers using RAGAS metrics
使用 RAGAS 评估学术论文 RAG 系统
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


def load_dataset(json_path):
    """Load RAGAS dataset from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to RAGAS format (使用 RAGAS 0.2+ 标准字段名)
    dataset_dict = {
        "user_input": [item["user_input"] for item in data],
        "response": [item["response"] for item in data],
        "retrieved_contexts": [item["retrieved_contexts"] for item in data],
        "reference": [item["reference"] for item in data],
    }

    return Dataset.from_dict(dataset_dict), data


def calculate_average(scores_list):
    """Calculate average from a list, filtering out None and NaN values"""
    valid_scores = [s for s in scores_list if s is not None and s == s]  # s == s filters NaN
    if not valid_scores:
        return None
    return sum(valid_scores) / len(valid_scores)


def evaluate_by_category(dataset_json, result):
    """Evaluate performance breakdown by question category"""
    categories = {}

    try:
        # 获取 DataFrame 格式的详细结果
        df = result.to_pandas()
        
        for i, item in enumerate(dataset_json):
            cat = item['category']
            if cat not in categories:
                categories[cat] = {
                    'faithfulness': [],
                    'answer_relevancy': [],
                    'context_precision': [],
                    'context_recall': [],
                }

            # 从 DataFrame 中获取每个样本的分数
            for metric in categories[cat].keys():
                if metric in df.columns and i < len(df):
                    score = df.iloc[i][metric]
                    if score is not None and score == score:  # 检查非 NaN
                        categories[cat][metric].append(score)

    except Exception as e:
        print(f"Warning: Could not get per-sample scores: {e}")
        return

    # Calculate averages
    print(f"\n{'='*60}")
    print("Performance Breakdown by Question Category")
    print(f"{'='*60}\n")

    for cat, metrics in categories.items():
        print(f"{cat.upper()} Questions:")
        for metric, scores in metrics.items():
            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {metric:20s}: {avg:.4f} (n={len(scores)})")
        print()


def main():
    """Main evaluation function"""
    print(f"{'='*60}")
    print("RAGAS Evaluation for Academic Papers")
    print(f"{'='*60}\n")

    # Load dataset
    dataset_path = project_root / "src/test/test_data/academic_ragas_dataset.json"
    print(f"Loading dataset: {dataset_path}")

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run 'python src/test/generate_academic_ragas_dataset.py' first")
        return

    dataset, dataset_json = load_dataset(dataset_path)
    print(f"✓ Loaded {len(dataset)} samples\n")

    # 打印数据集字段验证
    print("Dataset columns:", dataset.column_names)
    print(f"Sample user_input: {dataset['user_input'][0][:50]}...")
    print()

    # Initialize Mistral AI models
    print("Initializing Mistral AI models...")
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0)
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    print("✓ Models initialized\n")

    # Run RAGAS evaluation
    print("Running RAGAS evaluation...")
    print("This may take several minutes...\n")

    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings,
        )

        # Print overall results
        print(f"\n{'='*60}")
        print("Overall RAGAS Evaluation Results")
        print(f"{'='*60}\n")

        metrics_info = {
            'faithfulness': 'How factually accurate is the answer based on context?',
            'answer_relevancy': 'How relevant is the answer to the question?',
            'context_precision': 'Are the retrieved contexts relevant?',
            'context_recall': 'Are all relevant contexts retrieved?',
        }

        result_for_save = {}

        # 获取 DataFrame 以计算平均分
        df = result.to_pandas()

        for metric_name in metrics_info.keys():
            try:
                if metric_name in df.columns:
                    # 从 DataFrame 列计算平均值
                    scores = df[metric_name].tolist()
                    avg_score = calculate_average(scores)
                    
                    if avg_score is not None:
                        result_for_save[metric_name] = avg_score
                        print(f"{metric_name:20s}: {avg_score:.4f}")
                        print(f"  → {metrics_info[metric_name]}")
                        print()
                    else:
                        print(f"Warning: {metric_name} has no valid scores")
                else:
                    print(f"Warning: {metric_name} not found in result columns")
            except Exception as e:
                print(f"Warning: Error processing {metric_name}: {e}")

        # Evaluate by category
        evaluate_by_category(dataset_json, result)

        # Save results
        result_path = project_root / "src/test/test_data/academic_ragas_results.json"

        with open(result_path, 'w') as f:
            json.dump(result_for_save, f, indent=2)
        print(f"✓ Results saved to: {result_path}")

        # Performance interpretation
        print(f"\n{'='*60}")
        print("Performance Interpretation")
        print(f"{'='*60}\n")

        if len(result_for_save) == 0:
            print("⚠ No metrics were successfully evaluated!")
            print("\nTroubleshooting tips:")
            print("1. Check if your dataset fields match RAGAS 0.2+ format:")
            print("   - user_input, response, retrieved_contexts, reference")
            print("2. Verify your RAGAS version: pip show ragas")
            print("3. Check if LLM and embeddings are working correctly")
            return

        overall_avg = sum(result_for_save.values()) / len(result_for_save)

        if overall_avg >= 0.8:
            print("✓ EXCELLENT: Your RAG system performs very well on academic papers!")
        elif overall_avg >= 0.6:
            print("✓ GOOD: Your RAG system has solid performance with room for improvement.")
        elif overall_avg >= 0.4:
            print("⚠ MODERATE: Consider improving retrieval quality or answer generation.")
        else:
            print("⚠ NEEDS IMPROVEMENT: Significant optimization needed.")

        print(f"\nOverall Average Score: {overall_avg:.4f}")

        # Recommendations
        print(f"\n{'='*60}")
        print("Recommendations")
        print(f"{'='*60}\n")

        context_precision_score = result_for_save.get('context_precision', 0)
        context_recall_score = result_for_save.get('context_recall', 0)
        faithfulness_score = result_for_save.get('faithfulness', 0)
        answer_rel = result_for_save.get('answer_relevancy', 0)

        if context_precision_score < 0.6:
            print("• Low context precision: Consider improving your retrieval strategy")
            print("  - Try adjusting chunk size (currently in config.yaml)")
            print("  - Experiment with different embedding models")
            print("  - Increase top_k for more context\n")

        if context_recall_score < 0.6:
            print("• Low context recall: Not all relevant information is being retrieved")
            print("  - Increase top_k in retrieval configuration")
            print("  - Consider using MMR (Maximum Marginal Relevance) search\n")

        if faithfulness_score < 0.7:
            print("• Low faithfulness: Model is hallucinating or extrapolating")
            print("  - Adjust LLM prompt to emphasize using only provided context")
            print("  - Consider lowering temperature for more deterministic outputs\n")

        if answer_rel < 0.7:
            print("• Low answer relevancy: Answers are off-topic")
            print("  - Improve prompt engineering")
            print("  - Ensure retrieved contexts are relevant\n")

        if overall_avg >= 0.6:
            print("✓ No critical issues detected. Consider fine-tuning for further improvements.")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        # 额外调试信息
        print("\n" + "="*60)
        print("Debug Information")
        print("="*60)
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset columns: {dataset.column_names if hasattr(dataset, 'column_names') else 'N/A'}")
        print(f"First sample keys: {list(dataset_json[0].keys()) if dataset_json else 'N/A'}")


if __name__ == "__main__":
    main()