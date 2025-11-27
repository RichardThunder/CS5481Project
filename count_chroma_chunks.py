import os
import sys
from collections import Counter

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vector_store import VectorStoreManager

def main():
    try:
        print("Initializing VectorStoreManager...")
        manager = VectorStoreManager()
        
        print("Loading vector store...")
        vector_store = manager.load_vector_store()
        
        # Access the underlying Chroma collection
        # Note: _collection is technically internal, but commonly accessed for direct DB stats
        collection = vector_store._collection
        
        # 1. Count chunks
        count = collection.count()
        print(f"\n=== Data Statistics ===")
        print(f"Total number of chunks: {count}")
        
        if count == 0:
            print("The collection is empty.")
            return

        # 2. Get a sample to check embedding dimension and metadata
        # peek() returns a dictionary with keys like 'ids', 'embeddings', 'metadatas', 'documents'
        sample = collection.peek(limit=1)
        
        if sample['embeddings'] and len(sample['embeddings']) > 0:
            dim = len(sample['embeddings'][0])
            print(f"Embedding dimension: {dim}")
        else:
            print("Embedding dimension: N/A (no embeddings found in sample)")
            
        # 3. Source statistics (using get() to fetch metadata for all items is safe for moderate sizes)
        # For very large datasets, this might be slow, but for a local RAG it's usually fine.
        print("\nAnalyzing sources...")
        
        # Fetch all metadata
        all_data = collection.get(include=['metadatas'])
        metadatas = all_data['metadatas']
        
        sources = []
        for m in metadatas:
            if m and 'source' in m:
                sources.append(m['source'])
        
        source_counts = Counter(sources)
        
        print(f"Number of unique sources: {len(source_counts)}")
        print("\nChunks per source:")
        for source, num in source_counts.most_common():
            print(f"  - {source}: {num} chunks")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
