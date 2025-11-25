"""
Download and organize AI research papers from arXiv
从 arXiv 批量下载和组织 AI 论文
"""
import os
import arxiv
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent
documents_dir = project_root / "documents"

# 论文分类和 arXiv IDs
PAPER_CATEGORIES = {
    "rag": [
        ("2005.11401", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
        ("2310.11511", "Self-RAG: Learning to Retrieve, Generate and Critique"),
        ("2312.10997", "RAFT: Adapting Language Model to Domain Specific RAG"),
        ("2404.16130", "Graph RAG: Unlocking LLM discovery"),
    ],
    "llm": [
        ("2307.09288", "Llama 2: Open Foundation and Fine-Tuned Chat Models"),
        ("2310.06825", "Mistral 7B"),
        ("2203.02155", "Training language models to follow instructions"),
    ],
    "knowledge_graph": [
        ("2308.08998", "Knowledge Graphs for Enhanced LLM Reasoning"),
        ("1609.02907", "Semi-Supervised Classification with Graph Convolutional Networks"),
    ],
    "evaluation": [
        ("2309.15217", "RAGAS: Automated Evaluation of RAG"),
        ("2307.03109", "Evaluating Large Language Models: A Survey"),
    ],
}


def download_paper(arxiv_id: str, title: str, category: str):
    """Download a single paper from arXiv"""
    print(f"\nDownloading: {title}")
    print(f"  arXiv ID: {arxiv_id}")

    # 创建类别目录
    category_dir = documents_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    output_path = category_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    if output_path.exists():
        print(f"  ✓ Already exists: {output_path.name}")
        return

    try:
        # 搜索论文
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())

        # 下载 PDF
        paper.download_pdf(dirpath=str(category_dir), filename=output_path.name)
        print(f"  ✓ Downloaded: {output_path.name}")

    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    """Main function"""
    print("="*60)
    print("AI Research Papers Downloader")
    print("="*60)

    # 创建 documents 目录
    documents_dir.mkdir(exist_ok=True)

    total = sum(len(papers) for papers in PAPER_CATEGORIES.values())
    current = 0

    # 下载所有论文
    for category, papers in PAPER_CATEGORIES.items():
        print(f"\n{'='*60}")
        print(f"Category: {category.upper()}")
        print(f"{'='*60}")

        for arxiv_id, title in papers:
            current += 1
            print(f"\nProgress: {current}/{total}")
            download_paper(arxiv_id, title, category)

    # 统计
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}\n")

    for category in PAPER_CATEGORIES.keys():
        category_dir = documents_dir / category
        if category_dir.exists():
            pdf_count = len(list(category_dir.glob("*.pdf")))
            print(f"{category:20s}: {pdf_count} papers")

    total_pdfs = len(list(documents_dir.rglob("*.pdf")))
    print(f"\n{'Total':20s}: {total_pdfs} papers")
    print(f"\nDocuments location: {documents_dir}")


if __name__ == "__main__":
    main()
