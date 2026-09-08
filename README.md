# Website-Grounded RAG Agent

A production-quality Retrieval-Augmented Generation (RAG) system built for the MyAdvice AI Engineer Assessment. The agent crawls a publicly accessible website, builds a searchable knowledge base, and answers natural-language questions grounded exclusively in retrieved website content.

---

## What it does

1. Crawls a target website via a BFS crawler (depth-limited, same-domain)
2. Extracts and chunks content with header-aware splitting so section context is preserved
3. Indexes chunks in a hybrid store: dense embeddings (ChromaDB) and keyword-based (BM25)
4. At query time, expands the user question into 3 paraphrases and retrieves candidates for all 4
5. Merges candidates via Reciprocal Rank Fusion, then reranks with a cross-encoder
6. Routes through an answerability gate: answers with cited sources or refuses when evidence is insufficient
7. Tracks latency, token usage, and cost per query

---

## Tech Stack

| Layer | Library |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM | Gemini 3.6 Flash via LangChain |
| Embeddings | BAAI/bge-large-en-v1.5 (HuggingFace, free, local) |
| Vector store | ChromaDB |
| Keyword index | BM25 (rank_bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Crawling | requests + BeautifulSoup |

---

## Setup

### Requirements

- Python 3.10+
- Google Colab (recommended) or local Jupyter with GPU/CPU

### Environment Variables

Copy `.env.example` and fill in your key:

```
cp .env.example .env
```

| Variable | Source |
|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/) |

In Colab, add the key via the Secrets panel (key icon in the left sidebar) instead of a `.env` file.

### Installation

All dependencies are installed by running the first notebook cell:

```bash
pip install langchain langchain-google-genai langchain-chroma \
    langchain-text-splitters langchain-huggingface \
    langgraph beautifulsoup4 requests tiktoken chromadb \
    pandas matplotlib tabulate rank_bm25 sentence-transformers
```

---

## How to Run

1. Upload `Website_Grounded_RAG_Agent.ipynb` to Google Colab
2. Add `GEMINI_API_KEY` in Colab Secrets
3. Run all cells (Runtime > Run all)
4. The evaluation summary prints automatically in Section 15

> **Note:** BGE-large downloads approximately 1.3 GB on first run. Allow 5 minutes for the initial embedding build.

---

## Architecture

```
INGESTION
  Website -> BFS Crawler -> HTML Parser -> Header-aware chunker
          -> Chunk enrichment ([Title > Section] prepended)
          -> BGE-large embeddings -> ChromaDB
                                  -> BM25 index

QUERY
  Question -> Query Expansion (LLM, 3 paraphrases)
           -> Dense retrieval + BM25 retrieval (top-50 each, per query)
           -> RRF merge -> top-50 unique candidates
           -> Cross-encoder reranker -> top-5
           -> Answerability gate
                 No  -> Refuse with standard message
                 Yes -> Gemini LLM -> Answer + source URLs + latency trace
```

---

## Key Technical Decisions

### Embedding model: BGE-large-en-v1.5 over all-MiniLM-L6-v2
MiniLM (384-dim) is lexically sensitive; paraphrased queries fail to match. BGE-large (1024-dim, MTEB top-tier) is trained specifically for retrieval and handles semantic variation correctly. It runs locally at zero API cost.

### Multi-query expansion
A single query vocabulary may not match the corpus. The LLM generates 3 paraphrases; retrieval runs for all 4 queries. The resulting candidate lists are merged via RRF.

### Hybrid retrieval (dense + BM25) with RRF
Dense retrieval handles semantic queries; BM25 handles exact identifiers (`__init__`, `asyncio.create_task()`). RRF ranks-fuses both lists without score calibration.

### Cross-encoder reranker
Embedding similarity scores are coarse. A cross-encoder jointly scores each (question, chunk) pair, providing much more precise ranking before the LLM call.

### Answerability gate
If no chunk passes the reranker threshold, the system refuses rather than hallucinating. The threshold is set to -3.0 (calibrated on the eval set) to distinguish genuinely relevant chunks from noise.

### False-premise correction (Rule 6 in prompt)
Adversarial questions contain incorrect assumptions (e.g. "Since lists are immutable..."). The system prompt explicitly instructs the LLM to correct the false premise before answering.

---

## Evaluation

25-question golden dataset covering 5 categories:

| Category | Description |
|---|---|
| Straightforward | Direct terminology questions |
| Paraphrased | Same concepts, different vocabulary |
| Multi-hop | Require combining information across topics |
| Adversarial | Contain false premises |
| Unanswerable | Out-of-scope for the crawled website |

Metrics computed: Answerability Accuracy, Recall@K, Precision@K, Avg Reranker Score, Faithfulness (LLM-as-judge on 5 samples).

---

## Cost Analysis

**Ingestion:** Free (local BGE-large embeddings, no API calls)

**Per query (Gemini 3.6 Flash pricing as of mid-2025):**
| Component | Rate |
|---|---|
| Input tokens | $0.30 per 1M tokens |
| Output tokens | $2.50 per 1M tokens |

| Scale | Estimated Cost |
|---|---|
| 100 queries | ~$0.06 |
| 1,000 queries | ~$0.60 |
| 10,000 queries | ~$6.00 |

Refused (unanswerable) queries cost $0 as the LLM is never called.

---

## Known Limitations

- Query expansion adds ~1-2 seconds latency per query
- BGE-large is ~1.3 GB; production deployments should use an ONNX quantized variant
- Recall@K is measured by source-domain proxy, not labelled ground-truth chunks
- The BFS crawler is limited to 50 pages by default; deep documentation sites may require a higher limit

---

## Repository Structure

```
Website_Grounded_RAG_Agent.ipynb   Main notebook (v4 - recommended)
README.md                             This file
.env.example                          Environment variable template
```
