# LLM-Powered E-commerce Analytics Chatbot  
Local LLM + FAISS + LangChain + Streamlit

## Overview
This project is a fully local, offline analytics assistant designed for exploring and understanding multi-table e-commerce data (Olist dataset).  
It uses local embeddings, vector search, and a local LLM to provide contextual insights, EDA, chart generation, and natural language querying without any external APIs or cloud dependencies.

The system is built for performance, privacy, and cost-efficiency, making it ideal for learning, demos, and portfolio projects.

---

## Key Features

### 1. Local LLM Reasoning
- Powered by Mistral or Llama3 models running through **Ollama**.
- No API keys, no cloud billing, fully offline.
- Natural language question answering.
- Multi-step reasoning through LangChain.

### 2. Semantic Search with Local Vectors
- Uses **Sentence Transformers** for text embeddings.
- Stores vectors in a **FAISS** index for fast similarity search.
- Enables retrieval-augmented generation (RAG) over customer reviews and product metadata.

### 3. E-commerce Dataset Integration
- Processes the Olist multi-table dataset:
  - Customers  
  - Orders  
  - Items  
  - Products  
  - Sellers  
  - Payments  
  - Reviews  
- Merges all tables into a single denormalized analytics-friendly table.
- Computes delivery KPIs, delay metrics, review insights, and more.

### 4. Streamlit Interactive UI
- Chat interface for natural language queries.
- KPI cards and summary metrics.
- Automatic chart generation.
- Ability to perform SQL-style analytical queries with natural language.

---

## Technology Stack

### Core
- Python  
- LangChain  
- Streamlit  
- Pandas  

### Local AI
- Ollama  
- Mistral or Llama3  
- Sentence Transformers  

### Vector Search
- FAISS  

---

## Project Structure

