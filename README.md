
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

- **app/**
  - `app.py` — Streamlit UI
  - `chatbot_engine.py` — Core logic for answering questions
  - `create_local_vectorstore.py` — Embedding + FAISS index builder
  - `prepare_data.py` — Multi-table data processing
  - `test_local_retrieval.py` — Debug/testing retrieval scripts

- **data_raw/**  
  - Original CSV files (ignored by Git)

- **data_processed/**  
  - Processed parquet/CSV files  
  - FAISS index files

- **notebook/**  
  - Optional data exploration notebooks

- **venv/**  
  - Python virtual environment

- `.gitignore`
- `README.md`


---


## Setup Instructions

### 1. Install Dependencies

Activate your virtual environment and install packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas matplotlib seaborn
pip install langchain langchain-community
pip install sentence-transformers faiss-cpu
pip install python-dotenv
```

---

### 2. Install Ollama and Pull a Model

Install Ollama from:
https://ollama.com

Then pull a local LLM:

```bash
ollama pull mistral
```

---

### 3. Prepare the Dataset

Place the Olist CSV files inside:

```
data_raw/
```

Run the data processing script:

```bash
python app/prepare_data.py
```

---

### 4. Create the Vector Store

```bash
python app/create_local_vectorstore.py
```

---

### 5. Run the Streamlit App

```bash
streamlit run app/app.py
```

The app will open in your browser at:

```
http://localhost:8501
```


## Example Queries

-   Why are customers unhappy with late deliveries?
    
-   Show me monthly revenue trends.
    
-   Find reviews related to defective products.
    
-   What is the average delivery time for electronics?
    
-   Plot average review score by product category.
    
-   Which states have the highest late delivery rate?


## Future Improvements

-   Add support for SQL query generation and execution.
    
-   Deploy a multi-model choice interface (Mistral, Llama3, Qwen).
    
-   Improve chart variety (heatmaps, boxplots, treemaps).
    
-   Add a knowledge graph for cross-entity insights.
    
-   Containerize the project with Docker
