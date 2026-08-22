# Implementation Plan - Voice-Enabled Low-Latency RAG Model (HH Goa 2026 Task 2)

Build a complete, modular, and production-grade Python repository in `c:\Users\Garvit\Projects\hh_goa` for low-latency, high-precision voice RAG conversational question answering targeting sub-200ms pipeline execution.

## Proposed Architecture & Component Overview

```
 ┌────────────────┐      ┌─────────────────┐      ┌──────────────────┐
 │ Audio Stream / │ ───► │  stt_client.py  │ ───► │  rag_harness.py  │
 │  Text Input    │      │ (Sarvam/Eleven) │      │  (Orchestrator)  │
 └────────────────┘      └─────────────────┘      └────────┬─────────┘
                                                           │
                                ┌──────────────────────────┼──────────────────────────┐
                                ▼                          ▼                          ▼
                      ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
                      │  guardrails.py   │       │   indexer.py &   │       │     Groq LLM     │
                      │ (Injection/Check)│       │  qdrant_client   │       │ (llama-3.1-8b)   │
                      └──────────────────┘       └──────────────────┘       └──────────────────┘
```

## User Review Required

> [!IMPORTANT]
> **API Keys & Model Selection**:
> The system is designed to use **Groq API** (`llama-3.1-8b-instant`) for ultra-low latency LLM inference and **Sarvam AI / ElevenLabs** for STT. Fallbacks are included so the pipeline works seamlessly out-of-the-box locally even without live API keys.

> [!TIP]
> **Sub-200ms Latency Optimizations**:
> - Uses local CPU/GPU optimized embeddings (`BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`).
> - In-memory Qdrant instance with cosine distance indexing.
> - Hierarchical child (128-token) to parent (512-token) chunk retrieval.
> - Asynchronous streaming STT & parallel retrieval verification.

---

## Proposed Files & Changes

### [NEW] [config.py](file:///c:/Users/Garvit/Projects/hh_goa/config.py)
Configuration management using `pydantic-settings` or `python-dotenv`:
- API Keys (`GROQ_API_KEY`, `SARVAM_API_KEY`, `ELEVENLABS_API_KEY`).
- Qdrant Vector DB settings (Collection name, vector dimension, HNSW parameters).
- Latency thresholds & chunking parameters (child chunk size: 128, parent chunk size: 512, similarity threshold: 0.40).
- Fallback mode toggles.

### [NEW] [chunking.py](file:///c:/Users/Garvit/Projects/hh_goa/chunking.py)
Multi-strategy document chunking engine:
- `SemanticChunker`: Sentence segmentation and cosine similarity grouping of adjacent sentence embeddings.
- `HierarchicalChunker`: Parent document (512 tokens) and child document (128 tokens) mapping.
- `SlidingWindowChunker`: Fixed token/char window with configurable overlap and rich metadata.
- Data structures for `Chunk` and `ParentChildDoc`.

### [NEW] [indexer.py](file:///c:/Users/Garvit/Projects/hh_goa/indexer.py)
Dataset loader and Qdrant vector indexer:
- Dataset loader for `ai4bharat/MSMARCO-Xl` with pre-bundled fallback samples.
- Vector store indexer for Qdrant (in-memory / local HNSW).
- Embedding generator wrapping `SentenceTransformer` with batched processing.
- Methods: `initialize_collection()`, `index_documents()`, `search()`.

### [NEW] [stt_client.py](file:///c:/Users/Garvit/Projects/hh_goa/stt_client.py)
Async STT Client Adapter:
- `BaseSTTClient` abstract interface.
- `SarvamSTTClient`: Async API integration for Sarvam AI speech-to-text.
- `ElevenLabsSTTClient`: Async streaming API for ElevenLabs speech-to-text.
- `MockStreamingSTTClient`: Simulated streaming buffer client for low-latency testing without live audio feeds.

### [NEW] [guardrails.py](file:///c:/Users/Garvit/Projects/hh_goa/guardrails.py)
Validation and safety guardrail engine:
- Pydantic V2 structured output model (`GuardrailResponse`).
- Prompt injection & toxic query regex filter.
- Grounding verification: Cosine similarity score evaluation against context & refusal fallback logic.

### [NEW] [rag_harness.py](file:///c:/Users/Garvit/Projects/hh_goa/rag_harness.py)
Main RAG orchestration engine:
- Async execution pipeline: STT -> Embedding -> Qdrant Search -> Parent Context Lookup -> Guardrail Pre-check -> Groq LLM Inference -> Grounding Post-check.
- Granular latency instrumentation (per-stage timer breakdown in milliseconds).
- Retry policy with exponential backoff on network failures (`tenacity`).

### [NEW] [benchmark.py](file:///c:/Users/Garvit/Projects/hh_goa/benchmark.py)
Performance & Latency Benchmark Harness:
- Test suite execution over multi-query benchmark datasets.
- Calculation of latency percentiles: **P50**, **P70**, **P100** (max latency).
- Breakdown analysis per stage and output validation summary.

### [NEW] [app.py](file:///c:/Users/Garvit/Projects/hh_goa/app.py)
FastAPI Server:
- REST API `/api/v1/query`, `/api/v1/index`, `/api/v1/benchmark`.
- WebSocket `/ws/voice` for live streaming audio chunk processing.

### [NEW] [requirements.txt](file:///c:/Users/Garvit/Projects/hh_goa/requirements.txt)
Dependencies: `fastapi`, `uvicorn`, `qdrant-client`, `datasets`, `sentence-transformers`, `groq`, `pydantic`, `pydantic-settings`, `httpx`, `websockets`, `python-dotenv`, `tenacity`, `numpy`, `torch`.

### [NEW] [test_pipeline.py](file:///c:/Users/Garvit/Projects/hh_goa/test_pipeline.py)
End-to-end integration test runner to verify indexing, retrieval, guardrails, and benchmark metrics.

### [NEW] [README.md](file:///c:/Users/Garvit/Projects/hh_goa/README.md)
Comprehensive documentation detailing setup, configuration, architecture, and benchmark execution.

---

## Verification Plan

### Automated Tests
1. `pytest` test suite / `python test_pipeline.py`:
   - Tests all 3 chunking strategies (`SemanticChunker`, `HierarchicalChunker`, `SlidingWindowChunker`).
   - Verifies Qdrant index creation, parent-child retrieval, and grounding score checks.
   - Tests guardrails for prompt injection and out-of-grounding context queries.
   - Validates latency measurement and metrics output format.

2. Benchmark Run:
   - Run `python benchmark.py --queries 20` to verify latency percentiles (P50, P70, P100) and ensure sub-200ms target tracking.

### Manual Verification
1. Start FastAPI server via `uvicorn app:app --port 8000`.
2. Perform sample HTTP POST query to `/api/v1/query`.
3. Check WebSocket `/ws/voice` streaming response.
