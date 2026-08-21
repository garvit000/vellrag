# Voice-Enabled Low-Latency RAG Agent (HH Goa 2026 Task 2)

Low-latency, high-precision Voice RAG Agent built for conversational question answering with sub-200ms target execution times.

---

## Key Features

1. **Speech-to-Text (STT) Adapters**:
   - Asynchronous STT streaming client interface.
   - Sarvam AI STT adapter (Indic / Indian English).
   - ElevenLabs STT adapter.
   - Low-latency mock streaming STT client for offline benchmarking.

2. **Multi-Strategy Chunking**:
   - **Semantic Chunking**: Grouping sentences by cosine similarity threshold.
   - **Hierarchical (Parent-Document) Chunking**: Small 128-token child chunks for vector retrieval mapped to 512-token parent chunks for LLM context injection.
   - **Sliding Window Chunking**: Token window with configurable overlap and rich metadata.

3. **Fast Vector Indexing (Qdrant)**:
   - In-memory / fast local HNSW vector store using Cosine distance.
   - Dataset loader for `ai4bharat/MSMARCO-Xl` with pre-bundled fallback dataset.

4. **Pydantic V2 Guardrails & Structured Output**:
   - Schema enforcement: `{is_safe, context_grounded, confidence_score, refusal_reason, answer}`.
   - Pre-query safety & prompt injection detection.
   - Factual grounding check refusing queries when vector similarity < 0.40.
   - Voice conciseness filter removing Markdown, tables, bullets, and LaTeX.

5. **Groq LLM & Latency Instrumentation**:
   - Groq API integration using `llama-3.1-8b-instant`.
   - Granular execution timing per stage: `[STT -> Embedding -> Retrieval -> LLM -> Total]`.
   - Automatic exponential backoff retries on network errors (`tenacity`).

6. **Benchmarking Harness**:
   - Measures **P50**, **P70**, and **P100** latency percentiles.

---

## Directory Structure

```
hh_goa/
├── config.py          # Environment, thresholds, & API configurations
├── chunking.py        # Semantic, Hierarchical, & Sliding Window Chunkers
├── indexer.py         # MSMARCO-Xl loader & Qdrant HNSW vector indexer
├── stt_client.py      # Async STT Adapters (Sarvam, ElevenLabs, Mock)
├── guardrails.py      # Pydantic V2 schema & grounding verification
├── rag_harness.py     # Main Voice RAG engine & latency instrumentation
├── benchmark.py       # Latency benchmark harness (P50, P70, P100)
├── app.py             # FastAPI REST & WebSocket streaming server
├── test_pipeline.py   # Complete integration test runner
├── requirements.txt   # Dependencies
├── .env.example       # Environment key template
└── README.md          # Documentation
```

---

## Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` and set your API keys:
```bash
cp .env.example .env
```

---

## Running Benchmarks & Tests

### Run Pipeline Test Suite
```bash
python test_pipeline.py
```

### Run Latency Benchmark (P50, P70, P100)
```bash
python benchmark.py --iterations 1
```

---

## Running the Web Server

Start FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints
- **GET** `/health` - Health check & vector collection status.
- **POST** `/api/v1/query` - Send text queries to RAG pipeline.
- **POST** `/api/v1/index` - Trigger document re-indexing into Qdrant.
- **GET** `/api/v1/benchmark` - Trigger benchmark harness.
- **WebSocket** `/ws/voice` - Real-time streaming audio input & voice JSON responses.
