from src.vector_store import VectorStoreManager
import yaml

def debug_retrieval(query):
    print(f"Debug Retrieval for query: '{query}'")
    
    try:
        manager = VectorStoreManager()
        # We need to force load the vector store
        manager.load_vector_store()
        
        results = manager.similarity_search_with_score(query, k=10) # Get top 10 to see where Topic 1 is
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, (doc, score) in enumerate(results):
            content_preview = doc.page_content.replace('\n', ' ')[:100]
            print(f"{i+1}. Score: {score:.4f} | Content: {content_preview}...")
            
    except Exception as e:
        print(f"Error querying vector store: {e}")

if __name__ == "__main__":
    debug_retrieval("topic 1")

