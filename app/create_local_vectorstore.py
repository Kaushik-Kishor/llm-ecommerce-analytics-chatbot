import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(
            texts, 
            show_progress_bar=True,
            device="cpu"       # force CPU
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text], 
            device="cpu"       # force CPU
        )[0].tolist()


def make_vectorstore():
    print("Loading processed dataset...")
    df = pd.read_csv("../data_processed/olist_core_orders.csv")

   
    df = df[df["review_comment_message"].notna()].reset_index(drop=True)

    texts = df["review_comment_message"].tolist()

    print(f"Total reviews to embed: {len(texts)}")

   
    metadatas = df.to_dict(orient="records")

    
    print("Loading embedding model...")
    embedding = LocalEmbeddings("all-MiniLM-L6-v2")

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas
    )

    print("Saving vector store...")
    vectorstore.save_local("../data_processed/faiss_olist_reviews")

    print("Done! Vector store saved.")

if __name__ == "__main__":
    make_vectorstore()
