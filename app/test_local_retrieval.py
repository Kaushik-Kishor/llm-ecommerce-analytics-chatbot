from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings



class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, device="cpu").tolist()

    def embed_query(self, text):
        return self.model.encode([text], device="cpu")[0].tolist()


def run_query(query):
    print("Loading FAISS index...")
    db = FAISS.load_local(
        "../data_processed/faiss_olist_reviews",
        LocalEmbeddings("all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )

    print(f"\nSearching for: {query}\n")
    docs = db.similarity_search(query, k=3)

    print("Top retrieved reviews:\n")
    for i, d in enumerate(docs, 1):
        print(f"--- Review {i} ---")
        print(d.page_content[:500], "\n")

    
    context = "\n\n".join([d.page_content for d in docs])

    print("\nGenerating AI answer...\n")
    llm = Ollama(model="mistral")

    prompt = f"""
You are an analytics assistant for an e-commerce company.

Use the following customer reviews to answer the question.

Reviews:
{context}

Question: {query}

Give a short, clear insight:
"""

    response = llm.invoke(prompt)
    print("AI Answer:\n")
    print(response)


if __name__ == "__main__":
    run_query("Why are customers complaining about late deliveries?")
