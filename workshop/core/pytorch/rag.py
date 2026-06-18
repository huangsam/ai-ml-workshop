"""Retrieval-Augmented Generation (RAG) using PyTorch, Transformers (MiniLM and FLAN-T5), and NumPy."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

# --- 1. LOCAL KNOWLEDGE BASE (CORPUS) ---
DOCUMENTS = [
    "Antigravity is an advanced agentic AI coding assistant designed by the Google DeepMind team, specializing in pair programming and workspace refactoring.",
    "The AI/ML Workshop is a sandbox application developed in 2026 to help users learn machine learning concepts interactively through live SSE telemetry.",
    "The Project Triton is a top-secret research initiative by DeepMind focusing on ocean temperature modeling using specialized spatial-temporal graph neural networks.",
    "The first mechanical computer, invented by Charles Babbage in 1822, was called the Difference Engine and was designed to calculate mathematical tables.",
    "The term 'Machine Learning' was coined in 1959 by Arthur Samuel, an American pioneer in computer gaming and artificial intelligence.",
]

DOC_LABELS = ["Antigravity Assistant", "AI/ML Workshop App", "Project Triton", "Difference Engine (1822)", "Arthur Samuel ML (1959)"]


def mean_pooling(model_output, attention_mask):
    """Mean Pooling: computes sentence embedding from token embeddings taking attention mask into account."""
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings(texts: list[str], tokenizer, model, device: str = "cpu") -> np.ndarray:
    """Computes L2-normalized sentence embeddings using MiniLM."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = mean_pooling(model_output, inputs["attention_mask"])
    # Normalize embeddings to unit vectors for easy cosine similarity computation (dot product)
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized_embeddings.cpu().numpy()


def main(hook=None, config=None):
    """Entry point for the Retrieval-Augmented Generation (RAG) lesson."""
    from workshop.utils import get_device
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    query_index = int(config.get("query_index", 1))
    queries = {
        1: "Who designed the coding assistant Antigravity?",
        2: "What is the purpose of the AI/ML Workshop?",
        3: "What does Project Triton research?",
        4: "Who invented the Difference Engine mechanical computer?",
        5: "Who coined the term Machine Learning and when?",
    }
    query = queries.get(query_index, queries[1])
    top_k = int(config.get("top_k", 2))
    similarity_threshold = float(config.get("similarity_threshold", 0.0))

    device = get_device()
    print("Retrieval-Augmented Generation (RAG) Pipeline")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"User Query: '{query}'")
    print(f"Top-K Retrieve: {top_k}")
    print(f"Similarity Threshold: {similarity_threshold}")
    print()

    # --- Step 1: Load Models ---
    if hook.is_cancelled():
        return
    hook.update_stage("Loading Models", 10)
    print("Loading MiniLM embedding model and FLAN-T5-small generation model...")

    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    gen_model_name = "google/flan-t5-small"

    embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
    embed_model.eval()

    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(device)
    gen_model.eval()

    # --- Step 2: Encode Documents ---
    if hook.is_cancelled():
        return
    hook.update_stage("Encoding Documents", 30)
    print(f"Generating embeddings for {len(DOCUMENTS)} documents in the corpus...")
    doc_embeddings = get_embeddings(DOCUMENTS, embed_tokenizer, embed_model, device)

    # --- Step 3: Process Query ---
    if hook.is_cancelled():
        return
    hook.update_stage("Processing Query", 45)
    print(f"Generating embedding for query: '{query}'...")
    query_embedding = get_embeddings([query], embed_tokenizer, embed_model, device)[0]

    # --- Step 4: Retrieval & Search ---
    if hook.is_cancelled():
        return
    hook.update_stage("Retrieval & Search", 60)
    print("Searching the local vector space via cosine similarity...")

    # Dot product of normalized embeddings is identical to cosine similarity
    similarities = np.dot(doc_embeddings, query_embedding)

    # Sort in descending order
    sorted_indices = np.argsort(similarities)[::-1]

    # Filter by similarity threshold
    retrieved_indices = [idx for idx in sorted_indices if similarities[idx] >= similarity_threshold]
    top_indices = retrieved_indices[:top_k]

    print("\nSearch Results:")
    for rank, idx in enumerate(top_indices):
        print(f"Rank {rank + 1}: [Sim: {similarities[idx]:.4f}] '{DOCUMENTS[idx]}' ({DOC_LABELS[idx]})")
    print()

    # Construct augmented context
    retrieved_docs = [DOCUMENTS[idx] for idx in top_indices]
    if retrieved_docs:
        context_str = "\n".join([f"- {doc}" for doc in retrieved_docs])
    else:
        context_str = "No relevant documents found."

    # --- Step 5: Generation Without Context (Hallucination baseline) ---
    if hook.is_cancelled():
        return
    hook.update_stage("Generation Without Context", 75)
    print("Generating answer WITHOUT retrieval context (zero-shot baseline)...")

    prompt_no_context = f"Question: {query} Answer:"
    input_ids_no = gen_tokenizer(prompt_no_context, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs_no = gen_model.generate(input_ids_no, max_new_tokens=60, temperature=0.2, do_sample=False)
    answer_no_context = gen_tokenizer.decode(outputs_no[0], skip_special_tokens=True)

    print(f"Prompt Sent: '{prompt_no_context}'")
    print(f"Baseline Answer: '{answer_no_context}'")
    print()

    # --- Step 6: Generation With Context (RAG grounded) ---
    if hook.is_cancelled():
        return
    hook.update_stage("Generation With Context", 85)
    print("Generating answer WITH retrieved context (grounded RAG)...")

    prompt_with_context = (
        "Answer the question using ONLY the provided context below. "
        "If the context does not contain the answer, reply 'Based on the context, I do not know.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    input_ids_with = gen_tokenizer(prompt_with_context, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs_with = gen_model.generate(input_ids_with, max_new_tokens=60, temperature=0.2, do_sample=False)
    answer_with_context = gen_tokenizer.decode(outputs_with[0], skip_special_tokens=True)

    print(f"Prompt Sent:\n---\n{prompt_with_context}\n---")
    print(f"Grounded Answer: '{answer_with_context}'")
    print()

    # Send final answers as metrics for display
    hook.update_metrics(
        {
            "query": query,
            "retrieved_count": len(top_indices),
            "top_similarity": float(similarities[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0,
            "baseline_answer_snippet": answer_no_context[:60] + "..." if len(answer_no_context) > 60 else answer_no_context,
            "grounded_answer_snippet": answer_with_context[:60] + "..." if len(answer_with_context) > 60 else answer_with_context,
        }
    )

    # --- Step 7: Visualization ---
    if hook.is_cancelled():
        return
    hook.update_stage("Visualization", 90)

    # Plot 1: Cosine Similarity Scores Bar Chart
    plt.figure(figsize=(7, 4.5))

    # Sort docs for plotting
    plot_indices = np.argsort(similarities)  # Ascending order for horizontal bar chart
    sorted_labels = [DOC_LABELS[i] for i in plot_indices]
    sorted_sims = [similarities[i] for i in plot_indices]
    sorted_colors = ["#10b981" if i in top_indices else "#6b7280" for i in plot_indices]

    plt.barh(range(len(DOCUMENTS)), sorted_sims, color=sorted_colors, height=0.6)
    plt.yticks(range(len(DOCUMENTS)), sorted_labels)
    plt.xlabel("Cosine Similarity Score")
    plt.title("Query Cosine Similarity to Document Corpus (Top-K Green)", fontweight="bold")
    plt.xlim(0, 1.0)

    for i, val in enumerate(sorted_sims):
        plt.text(val + 0.02, i, f"{val:.4f}", va="center", fontweight="bold")

    plt.tight_layout()
    hook.save_plot("rag_similarity_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Embedding Space 2D PCA Scatter Plot
    # Stack document and query embeddings: shape (6, 384)
    all_embeddings = np.vstack([doc_embeddings, query_embedding.reshape(1, -1)])

    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(all_embeddings)

    plt.figure(figsize=(7, 6))

    # Plot corpus documents
    for idx in range(len(DOCUMENTS)):
        is_retrieved = idx in top_indices
        color = "#10b981" if is_retrieved else "#9ca3af"
        marker = "o" if is_retrieved else "o"
        size = 120 if is_retrieved else 60
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c=color, s=size, marker=marker, label="Retrieved Context" if idx == top_indices[0] else "")
        plt.text(embeddings_2d[idx, 0] + 0.02, embeddings_2d[idx, 1] + 0.02, DOC_LABELS[idx], fontsize=9, alpha=0.9 if is_retrieved else 0.6)

    # Plot query
    plt.scatter(embeddings_2d[5, 0], embeddings_2d[5, 1], c="#ef4444", s=180, marker="*", label="User Query")
    plt.text(embeddings_2d[5, 0] + 0.02, embeddings_2d[5, 1] + 0.02, "QUERY", fontsize=10, fontweight="bold", color="#ef4444")

    # Draw dashed lines connecting query to retrieved documents
    for idx in top_indices:
        plt.plot([embeddings_2d[5, 0], embeddings_2d[idx, 0]], [embeddings_2d[5, 1], embeddings_2d[idx, 1]], "k--", alpha=0.5)
        mid_x = (embeddings_2d[5, 0] + embeddings_2d[idx, 0]) / 2
        mid_y = (embeddings_2d[5, 1] + embeddings_2d[idx, 1]) / 2
        plt.text(mid_x, mid_y, f"sim={similarities[idx]:.2f}", fontsize=8, bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"))

    plt.title("2D Projection (PCA) of Semantic Embedding Space", fontweight="bold")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    hook.save_plot("rag_embedding_space.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
