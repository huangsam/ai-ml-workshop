# Project 2: Semantic Search Engine with RAG

## 🎯 Goal
Build a Question-Answering (QA) application that uses a custom set of documents (the Knowledge Base) to ground the LLM's answers.

## 🛠️ Technology Stack
* **Orchestration:** LangChain
* **Embeddings:** Hugging Face `SentenceTransformers` (via LangChain's integration) or OpenAI Embeddings API
* **Vector Store:** ChromaDB or FAISS (local vector databases)
* **LLM:** Any LLM API (e.g., OpenAI, Anthropic) or a local Hugging Face LLM (e.g., Llama 3 via `transformers`).

## ⚙️ Steps & Milestones

### 1. Data Ingestion & Splitting
1.  **Loaders:** Use a LangChain **Document Loader** (e.g., `DirectoryLoader` for a folder of text files) to ingest your raw data.
2.  **Splitters:** Use a LangChain **Text Splitter** (e.g., `RecursiveCharacterTextSplitter`) to break large documents into small, manageable chunks suitable for vector search.

### 2. Embedding & Indexing
1.  **Embeddings:** Initialize a LangChain **Embeddings** model (e.g., `HuggingFaceEmbeddings`).
2.  **Vector Store:** Create an index by passing the document chunks and the embeddings model to a local **Vector Store** (e.g., `Chroma.from_documents()`). This generates and stores the vectors.

### 3. Retrieval and Chain Creation
1.  **Retriever:** Convert the Vector Store object into a **Retriever** object (`vectorstore.as_retriever()`).
2.  **Chain:** Construct a **Retrieval-Augmented Generation (RAG) Chain**. This chain handles:
    * Receiving the user's question.
    * Querying the Retriever for relevant text chunks.
    * Constructing a final **prompt** that includes the question AND the retrieved context.
3.  **LLM Call:** Send the final prompt to the LLM to generate a grounded answer.

[Image of retrieval augmented generation (RAG) workflow]


### 💡 Key Learning Points
* The difference between **text** and **embeddings (vectors)**.
* The three core RAG components: **Loader/Splitter**, **Embeddings/VectorStore**, and **Retriever**.
* How to use a chain to ensure the LLM only answers based on the provided context.
