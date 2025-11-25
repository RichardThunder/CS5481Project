"""
Vector Store Database Testing Suite
Tests for ChromaDB vector database operations, data integrity, and retrieval quality
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(dotenv_path=project_root / ".env")

from src.vector_store import VectorStoreManager
from langchain_core.documents import Document


class VectorStoreTestSuite:
    """Comprehensive test suite for vector database operations"""

    def __init__(self, config_path: str = None):
        """Initialize the test suite with a test configuration"""
        if config_path is None:
            config_path = str(project_root / "config.yaml")

        self.config_path = config_path
        self.test_db_path = str(project_root / "test_chroma_db")
        self.test_results = {}

    def setup_test_db(self):
        """Create a temporary test database"""
        print(f"\n{'='*60}")
        print("Setting up test database...")
        print(f"{'='*60}")

        # Clean up if exists
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)

        # Create test vector store manager
        self.vector_manager = VectorStoreManager(self.config_path)
        self.vector_manager.persist_directory = self.test_db_path

        print(f"✓ Test database directory: {self.test_db_path}")

    def teardown_test_db(self):
        """Clean up test database"""
        print(f"\n{'='*60}")
        print("Cleaning up test database...")
        print(f"{'='*60}")

        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
            print("✓ Test database removed")

    def create_test_documents(self) -> List[Document]:
        """Create sample documents for testing"""
        test_docs = [
            Document(
                page_content="GraphRAG combines graph-based knowledge retrieval with large language models for enhanced reasoning.",
                metadata={"source": "test_doc_1.txt", "topic": "GraphRAG", "type": "definition"}
            ),
            Document(
                page_content="Retrieval-Augmented Generation (RAG) improves LLM responses by incorporating external knowledge from document databases.",
                metadata={"source": "test_doc_2.txt", "topic": "RAG", "type": "definition"}
            ),
            Document(
                page_content="Vector databases use embeddings to store and retrieve semantically similar documents efficiently.",
                metadata={"source": "test_doc_3.txt", "topic": "vector_db", "type": "definition"}
            ),
            Document(
                page_content="ChromaDB is an open-source embedding database designed for AI applications with support for metadata filtering.",
                metadata={"source": "test_doc_4.txt", "topic": "chromadb", "type": "technology"}
            ),
            Document(
                page_content="Causal reasoning in AI involves understanding cause-and-effect relationships between events and entities.",
                metadata={"source": "test_doc_5.txt", "topic": "causal_reasoning", "type": "concept"}
            ),
            Document(
                page_content="Knowledge graphs structure information as entities and relationships, enabling complex queries and reasoning.",
                metadata={"source": "test_doc_6.txt", "topic": "knowledge_graph", "type": "concept"}
            ),
            Document(
                page_content="The GraphRAG-Causal framework uses Neo4j for storing causal knowledge graphs extracted from news articles.",
                metadata={"source": "test_doc_7.txt", "topic": "GraphRAG", "type": "implementation"}
            ),
            Document(
                page_content="Embedding models convert text into dense vector representations that capture semantic meaning.",
                metadata={"source": "test_doc_8.txt", "topic": "embeddings", "type": "technology"}
            ),
        ]
        return test_docs

    # ============================================================
    # Test 1: Database Creation and Persistence
    # ============================================================

    def test_database_creation(self) -> bool:
        """Test if vector database can be created and persisted"""
        print(f"\n{'='*60}")
        print("Test 1: Database Creation and Persistence")
        print(f"{'='*60}")

        try:
            test_docs = self.create_test_documents()

            # Create vector store
            self.vector_manager.create_vector_store(test_docs)

            # Check if directory exists
            if not os.path.exists(self.test_db_path):
                print("✗ FAILED: Database directory not created")
                return False

            # Check if ChromaDB files exist
            db_files = os.listdir(self.test_db_path)
            if len(db_files) == 0:
                print("✗ FAILED: No database files created")
                return False

            print(f"✓ PASSED: Database created with {len(test_docs)} documents")
            print(f"  Database files: {db_files}")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    # ============================================================
    # Test 2: Document Loading and Retrieval
    # ============================================================

    def test_document_loading(self) -> bool:
        """Test if documents can be loaded from persisted database"""
        print(f"\n{'='*60}")
        print("Test 2: Document Loading from Persisted Database")
        print(f"{'='*60}")

        try:
            # Create a new manager instance (simulating app restart)
            new_manager = VectorStoreManager(self.config_path)
            new_manager.persist_directory = self.test_db_path

            # Load existing database
            new_manager.load_vector_store()

            # Try to retrieve documents
            results = new_manager.similarity_search("GraphRAG", k=3)

            if len(results) == 0:
                print("✗ FAILED: No documents retrieved from loaded database")
                return False

            print(f"✓ PASSED: Successfully loaded database and retrieved {len(results)} documents")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    # ============================================================
    # Test 3: Similarity Search Accuracy
    # ============================================================

    def test_similarity_search(self) -> bool:
        """Test if similarity search returns relevant results"""
        print(f"\n{'='*60}")
        print("Test 3: Similarity Search Accuracy")
        print(f"{'='*60}")

        try:
            # Test queries with expected topics
            test_cases = [
                {
                    "query": "What is GraphRAG?",
                    "expected_topics": ["GraphRAG"],
                    "k": 3
                },
                {
                    "query": "How do vector databases work?",
                    "expected_topics": ["vector_db", "chromadb", "embeddings"],
                    "k": 3
                },
                {
                    "query": "Explain causal reasoning",
                    "expected_topics": ["causal_reasoning", "knowledge_graph"],
                    "k": 2
                },
            ]

            passed = 0
            total = len(test_cases)

            for i, test_case in enumerate(test_cases, 1):
                query = test_case["query"]
                expected_topics = test_case["expected_topics"]
                k = test_case["k"]

                print(f"\n  Test Case {i}:")
                print(f"  Query: {query}")
                print(f"  Expected topics: {expected_topics}")

                results = self.vector_manager.similarity_search(query, k=k)

                # Check if any expected topic is in the results
                retrieved_topics = [doc.metadata.get("topic") for doc in results]
                print(f"  Retrieved topics: {retrieved_topics}")

                # Success if at least one expected topic is in top results
                if any(topic in retrieved_topics for topic in expected_topics):
                    print(f"  ✓ Relevant document found")
                    passed += 1
                else:
                    print(f"  ✗ No relevant documents found")

            success_rate = passed / total
            print(f"\n{'='*60}")
            print(f"Similarity Search Success Rate: {success_rate:.1%} ({passed}/{total})")

            if success_rate >= 0.8:  # 80% threshold
                print("✓ PASSED: Similarity search is accurate")
                return True
            else:
                print("✗ FAILED: Similarity search accuracy below threshold")
                return False

        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ============================================================
    # Test 4: Search with Scores
    # ============================================================

    def test_similarity_search_with_scores(self) -> bool:
        """Test if similarity scores are returned and reasonable"""
        print(f"\n{'='*60}")
        print("Test 4: Similarity Search with Scores")
        print(f"{'='*60}")

        try:
            query = "What is GraphRAG and how does it work?"
            results = self.vector_manager.similarity_search_with_score(query, k=5)

            if len(results) == 0:
                print("✗ FAILED: No results returned")
                return False

            print(f"\n  Query: {query}")
            print(f"  Results:")

            # Check if scores are in descending order (lower is better in ChromaDB)
            scores = [score for _, score in results]
            is_sorted = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))

            for i, (doc, score) in enumerate(results, 1):
                print(f"\n  Rank {i}:")
                print(f"    Score: {score:.4f}")
                print(f"    Topic: {doc.metadata.get('topic')}")
                print(f"    Content: {doc.page_content[:80]}...")

            if is_sorted:
                print(f"\n✓ PASSED: Scores are properly ordered (ascending distance)")
                return True
            else:
                print(f"\n✗ FAILED: Scores are not properly ordered")
                return False

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    # ============================================================
    # Test 5: Metadata Filtering
    # ============================================================

    def test_metadata_filtering(self) -> bool:
        """Test if metadata filters work correctly"""
        print(f"\n{'='*60}")
        print("Test 5: Metadata Filtering")
        print(f"{'='*60}")

        try:
            # Test filtering by type
            filter_tests = [
                {"filter": {"type": "definition"}, "expected_min": 2},
                {"filter": {"type": "concept"}, "expected_min": 1},
                {"filter": {"topic": "GraphRAG"}, "expected_min": 1},
            ]

            passed = 0
            total = len(filter_tests)

            for i, test in enumerate(filter_tests, 1):
                filter_dict = test["filter"]
                expected_min = test["expected_min"]

                print(f"\n  Test Case {i}: Filter = {filter_dict}")

                results = self.vector_manager.similarity_search(
                    "Tell me about AI and machine learning",
                    k=10,
                    filter=filter_dict
                )

                print(f"  Retrieved {len(results)} documents")

                # Verify all results match the filter
                all_match = all(
                    all(doc.metadata.get(k) == v for k, v in filter_dict.items())
                    for doc in results
                )

                if all_match and len(results) >= expected_min:
                    print(f"  ✓ Filter working correctly")
                    passed += 1
                else:
                    print(f"  ✗ Filter not working as expected")

            if passed == total:
                print(f"\n✓ PASSED: All metadata filters working correctly")
                return True
            else:
                print(f"\n✗ FAILED: {total - passed} filter test(s) failed")
                return False

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    # ============================================================
    # Test 6: Incremental Document Addition
    # ============================================================

    def test_incremental_addition(self) -> bool:
        """Test if new documents can be added to existing database"""
        print(f"\n{'='*60}")
        print("Test 6: Incremental Document Addition")
        print(f"{'='*60}")

        try:
            # Add new documents
            new_docs = [
                Document(
                    page_content="Few-shot learning enables models to learn from limited examples.",
                    metadata={"source": "test_doc_9.txt", "topic": "few_shot", "type": "concept"}
                ),
                Document(
                    page_content="Prompt engineering is crucial for optimizing LLM performance.",
                    metadata={"source": "test_doc_10.txt", "topic": "prompting", "type": "technique"}
                ),
            ]

            # Count before addition
            results_before = self.vector_manager.similarity_search("AI techniques", k=100)
            count_before = len(results_before)

            print(f"  Documents before addition: {count_before}")

            # Add new documents
            self.vector_manager.add_documents(new_docs)

            # Count after addition
            results_after = self.vector_manager.similarity_search("AI techniques", k=100)
            count_after = len(results_after)

            print(f"  Documents after addition: {count_after}")

            if count_after >= count_before + len(new_docs):
                print(f"✓ PASSED: Successfully added {len(new_docs)} new documents")
                return True
            else:
                print(f"✗ FAILED: Document count not increased as expected")
                return False

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    # ============================================================
    # Test 7: Retriever Interface
    # ============================================================

    def test_retriever_interface(self) -> bool:
        """Test if the retriever interface works for LangChain integration"""
        print(f"\n{'='*60}")
        print("Test 7: Retriever Interface (LangChain Integration)")
        print(f"{'='*60}")

        try:
            retriever = self.vector_manager.as_retriever(k=3)

            # Test retrieval
            query = "What is GraphRAG?"
            results = retriever.invoke(query)

            if len(results) == 0:
                print("✗ FAILED: Retriever returned no results")
                return False

            print(f"✓ PASSED: Retriever interface working")
            print(f"  Retrieved {len(results)} documents for query: '{query}'")
            print(f"  First result topic: {results[0].metadata.get('topic')}")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    # ============================================================
    # Test 8: Database Statistics and Health Check
    # ============================================================

    def test_database_health(self) -> bool:
        """Test database health and statistics"""
        print(f"\n{'='*60}")
        print("Test 8: Database Health Check")
        print(f"{'='*60}")

        try:
            # Test various queries to ensure consistent behavior
            test_queries = [
                "GraphRAG",
                "vector database",
                "causal reasoning",
                "embeddings",
                "knowledge graph"
            ]

            all_healthy = True

            for query in test_queries:
                try:
                    results = self.vector_manager.similarity_search(query, k=3)
                    if len(results) == 0:
                        print(f"  ⚠ Warning: No results for query '{query}'")
                        all_healthy = False
                except Exception as e:
                    print(f"  ✗ Error with query '{query}': {e}")
                    all_healthy = False

            # Check database size
            if os.path.exists(self.test_db_path):
                db_size = sum(
                    os.path.getsize(os.path.join(self.test_db_path, f))
                    for f in os.listdir(self.test_db_path)
                    if os.path.isfile(os.path.join(self.test_db_path, f))
                )
                print(f"\n  Database size: {db_size / 1024:.2f} KB")

            if all_healthy:
                print(f"\n✓ PASSED: Database is healthy")
                return True
            else:
                print(f"\n⚠ WARNING: Some queries failed")
                return False

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    # ============================================================
    # Run All Tests
    # ============================================================

    def run_all_tests(self, cleanup: bool = True) -> Dict[str, bool]:
        """Run all tests and return results"""
        print(f"\n{'#'*60}")
        print("VECTOR STORE DATABASE TEST SUITE")
        print(f"{'#'*60}")

        # Setup
        self.setup_test_db()

        # Run tests
        tests = [
            ("Database Creation", self.test_database_creation),
            ("Document Loading", self.test_document_loading),
            ("Similarity Search", self.test_similarity_search),
            ("Search with Scores", self.test_similarity_search_with_scores),
            ("Metadata Filtering", self.test_metadata_filtering),
            ("Incremental Addition", self.test_incremental_addition),
            ("Retriever Interface", self.test_retriever_interface),
            ("Database Health", self.test_database_health),
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"\n✗ CRITICAL ERROR in {test_name}: {e}")
                results[test_name] = False

        # Print summary
        self.print_test_summary(results)

        # Cleanup
        if cleanup:
            self.teardown_test_db()

        return results

    def print_test_summary(self, results: Dict[str, bool]):
        """Print a summary of test results"""
        print(f"\n{'#'*60}")
        print("TEST SUMMARY")
        print(f"{'#'*60}\n")

        passed = sum(results.values())
        total = len(results)

        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"  {test_name:30s}: {status}")

        print(f"\n{'='*60}")
        print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print(f"{'='*60}")

        if passed == total:
            print("\n🎉 All tests passed! Vector database is working correctly.")
        else:
            print(f"\n⚠ {total - passed} test(s) failed. Please review the errors above.")

        # Save results
        self.save_test_results(results)

    def save_test_results(self, results: Dict[str, bool]):
        """Save test results to JSON file"""
        output_dir = project_root / "src/test/test_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "vector_store_test_results.json"

        test_data = {
            "test_results": results,
            "summary": {
                "total_tests": len(results),
                "passed": sum(results.values()),
                "failed": len(results) - sum(results.values()),
                "success_rate": f"{sum(results.values())/len(results)*100:.1f}%"
            }
        }

        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"\n📊 Test results saved to: {output_file}")


def main():
    """Main function to run the test suite"""
    tester = VectorStoreTestSuite()
    results = tester.run_all_tests(cleanup=True)

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
