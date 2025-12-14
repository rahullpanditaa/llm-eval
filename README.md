# LLM Evaluation Pipeline

This repository contains a **real-time LLM response evaluation pipeline** designed to automatically assess the reliability of AI-generated answers against multiple quality metrics.

The project focuses on **evaluation logic, scalability, and practical engineering tradeoffs**, rather than model fine-tuning or prompt optimization.

---

## ✨ What This Project Does

Given:

* A **chat conversation JSON** (containing user/assistant turns)
* A **context vectors JSON** (retrieved chunks from a vector database for the last user query)

The pipeline:

1. Generates an AI answer in real time using a local LLM
2. Measures **latency and estimated cost**
3. Evaluates the answer for:

   * **Relevance** to the user query
   * **Completeness** with respect to retrieved context
   * **Hallucination / factual grounding**
4. Outputs structured evaluation results (and persists them to JSON)

---

## 🧠 Evaluation Dimensions

### 1. Response Relevance

Measures how well the generated answer aligns with the user query.

**Signals used:**

* Semantic similarity (query ↔ answer)
* Lexical overlap (content-word overlap)

These signals are combined into a single relevance score with an explanatory note.

---

### 2. Response Completeness

Measures how much of the retrieved context is reflected in the generated answer.

**Approach:**

* Embed the answer and each retrieved context chunk
* Compute semantic similarity
* Count how many context chunks are "covered" by the answer

The score is calculated as:

```
covered_chunks / total_chunks
```

---

### 3. Hallucination / Factual Accuracy

Measures whether the answer contains claims **not supported by retrieved context**.

**Approach:**

* Split the answer into sentences (atomic claims)
* For each sentence, compute maximum semantic similarity against all context chunks
* Mark sentences below a support threshold as hallucinated

The hallucination score is the fraction of unsupported sentences.

---

## 🏗️ Architecture Overview

```
Input JSONs
   ↓
Input Extraction
   ↓
Prompt Construction
   ↓
LLM Answer Generation (Ollama - Mistral)
   ↓
Latency & Cost Measurement
   ↓
Relevance & Completeness Evaluation
   ↓
Hallucination Detection
   ↓
Structured Output (JSON)
```

Each step is **deterministic, modular, and independently testable**.

---

## ⚙️ Technologies Used

* **Python 3.12**
* **Ollama** (local LLM inference)
* **Mistral (instruction-tuned)**
* **SentenceTransformers** (semantic embeddings)
* **NLTK** (sentence tokenization)
* **uv** (dependency & environment management)

No external APIs are required.

---

## 🚀 Local Setup Instructions

### 1. Clone the repository

```bash
git clone <git@github.com:rahullpanditaa/llm-eval.git>
cd llm-eval
```

### 2. Create environment & install dependencies

```bash
uv sync
```

### 3. Install NLTK tokenizer data (one-time)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 4. Install Ollama and pull model

```bash
ollama pull mistral
```

### 5. Run the evaluation pipeline

```bash
uv run cli/llm_eval.py run-eval --conversation 1
```

Optional:

```bash
uv run cli/llm_eval.py run-eval --conversation 2 --k 5
```

---

## 📦 Output Format

The pipeline returns (and can optionally persist) a structured JSON output:

```json
{
  "input": {
    "conversation_id": 1,
    "user_query": "...",
    "num_context_chunks": 5
  },
  "generation": {
    "answer": "...",
    "latency_ms": 41234.5,
    "tokens": { "input": 2604, "output": 156, "total": 2760 },
    "estimated_cost": 0.0067
  },
  "evaluation": {
    "relevance": { "score": 0.66, "note": "..." },
    "completeness": { "score": 0.2, "note": "..." },
    "hallucination": { "score": 0.5, "note": "..." }
  }
}
```

---

## 📈 Latency & Cost Measurement

* **Latency** is measured using wall-clock timing around model inference
* **Token counts** are computed for input and output
* **Estimated cost** is calculated using configurable per-million-token pricing

This allows the pipeline to simulate **real-time evaluation constraints**.

---

## 🔍 Design Decisions & Tradeoffs

* **Local LLM (Ollama)** was used to avoid API dependencies and enable reproducibility
* **No LLM-as-a-judge** is used; all evaluation is deterministic and embedding-based
* **Semantic similarity** is favored over brittle keyword rules
* **Minimal heuristics** are used to ensure scalability and explainability

The goal is to evaluate **answer reliability**, not to optimize model quality.

---

## ⚡ Scaling Considerations

If run at **millions of conversations per day**:

* No additional LLM calls beyond answer generation
* Embeddings reused within each evaluation step
* Deterministic evaluation logic (O(n) per context chunk)
* No external API latency or cost
* Easy batching and parallelization opportunities

The pipeline is designed to remain **low-latency and cost-efficient** at scale.

---

## ✅ Status

* End-to-end pipeline implemented
* Real-time evaluation supported
* All task requirements satisfied

---

## 📌 Notes

This project intentionally prioritizes **clarity, correctness, and engineering judgment** over feature completeness or heavy optimization.
