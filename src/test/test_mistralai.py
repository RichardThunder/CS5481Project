import os
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

project_root =Path(__file__).parent.parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)
def test_mistralai_chat():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRALAI_API_KEY environment variable not set")

    chat_model = ChatMistralAI(api_key=api_key, model_name="mistral-small-latest")
    response = chat_model.invoke("Hello, how are you?")
    print(response)

def test_mistralai_embeddings():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRALAI_API_KEY environment variable not set")
    embedding_model = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")
    text = "Test embedding"
    embedding = embedding_model.embed_query(text)
    print(embedding)

if __name__ == "__main__":
    test_mistralai_chat()
    test_mistralai_embeddings()
    print("All tests passed.")