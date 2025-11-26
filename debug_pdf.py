
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def debug_pdf(file_path):
    print(f"Checking file: {file_path}")
    
    if not os.path.exists(file_path):
        print("File does not exist!")
        return

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages.")
    
    full_text = ""
    for i, doc in enumerate(documents):
        print(f"--- Page {i+1} Preview ---")
        print(doc.page_content[:200]) # Print first 200 chars of each page
        full_text += doc.page_content + "\n"

    target = "topic 1"
    if target.lower() in full_text.lower():
        print(f"\nFound '{target}' in text!")
    else:
        print(f"\nDid NOT find '{target}' in text. This is likely the issue (OCR needed?).")
        return

    # Check chunking
    print("\n--- Checking Chunking ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    found_in_chunks = 0
    for i, chunk in enumerate(chunks):
        if target.lower() in chunk.page_content.lower():
            print(f"\nMatch in Chunk {i}:")
            print("-" * 20)
            print(chunk.page_content)
            print("-" * 20)
            found_in_chunks += 1
            
    print(f"\n'{target}' found in {found_in_chunks} chunks.")

if __name__ == "__main__":
    debug_pdf("documents/CS5481_Project.pdf")
