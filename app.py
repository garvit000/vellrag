"""
FastAPI Web Server providing REST and WebSocket Endpoints for Voice RAG System.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from indexer import VectorIndexer, load_msmarco_xl_dataset
from chunking import HierarchicalChunker
from rag_harness import VoiceRAGEngine
from stt_client import SarvamSTTClient, ElevenLabsSTTClient, MockStreamingSTTClient
from guardrails import GuardrailResponse
from benchmark import LatencyBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceRAGApp")

app = FastAPI(
    title="HH Goa 2026 Voice RAG Agent API",
    description="Sub-200ms Low-Latency Voice-Enabled RAG System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine Instance
indexer_instance = VectorIndexer()
engine_instance = VoiceRAGEngine(indexer=indexer_instance)


class TextQueryRequest(BaseModel):
    query: str = Field(..., example="What is the target latency for voice RAG pipelines?")
    top_k: int = Field(default=2, ge=1, le=5)


class IndexRequest(BaseModel):
    num_documents: int = Field(default=5, ge=1, le=50)
    strategy: str = Field(default="hierarchical", example="hierarchical")


@app.on_event("startup")
async def startup_event():
    logger.info("Initializing vector index with sample MSMARCO documents on startup...")
    docs = load_msmarco_xl_dataset(limit=5)
    chunker = HierarchicalChunker(parent_size=512, child_size=128, overlap=32)

    parent_child_list = []
    for doc in docs:
        parent_child_list.extend(chunker.chunk(doc["doc_id"], doc["text"]))

    indexer_instance.index_parent_child_docs(parent_child_list)
    logger.info("Startup indexing complete.")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "collection": settings.QDRANT_COLLECTION,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "target_latency_ms": settings.TARGET_PIPELINE_LATENCY_MS
    }


@app.post("/api/v1/query")
async def query_text(req: TextQueryRequest):
    """REST Endpoint for text queries."""
    response, metrics = await engine_instance.process_query(req.query, top_k=req.top_k)
    return {
        "response": response.model_dump(),
        "latency_metrics": metrics.to_dict()
    }


@app.post("/api/v1/query/audio")
async def query_audio(file: UploadFile = File(...)):
    """REST Endpoint for audio file input."""
    audio_bytes = await file.read()
    response, metrics = await engine_instance.process_audio(audio_bytes)
    return {
        "response": response.model_dump(),
        "latency_metrics": metrics.to_dict()
    }


@app.post("/api/v1/index")
async def trigger_indexing(req: IndexRequest):
    """Trigger document re-indexing into Qdrant."""
    docs = load_msmarco_xl_dataset(limit=req.num_documents)
    chunker = HierarchicalChunker(parent_size=512, child_size=128, overlap=32)

    parent_child_list = []
    for doc in docs:
        parent_child_list.extend(chunker.chunk(doc["doc_id"], doc["text"]))

    indexer_instance.index_parent_child_docs(parent_child_list)
    return {"message": f"Successfully indexed {len(docs)} documents.", "chunks_indexed": len(parent_child_list)}


@app.get("/api/v1/benchmark")
async def run_benchmark(iterations: int = 1):
    """Run latency benchmark suite and return P50, P70, P100 metrics."""
    bm = LatencyBenchmark()
    summary = await bm.run_benchmark(iterations=iterations)
    return summary


@app.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio chunk streaming.
    Receives binary audio data chunks, processes through low-latency pipeline,
    and returns voice JSON responses.
    """
    await websocket.accept()
    logger.info("WebSocket voice stream connection opened.")

    try:
        audio_buffer = bytearray()
        while True:
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                audio_buffer.extend(data["bytes"])

                # When buffer exceeds ~16KB (~0.5s audio chunk)
                if len(audio_buffer) >= 16000:
                    response, metrics = await engine_instance.process_audio(bytes(audio_buffer))
                    await websocket.send_json({
                        "event": "voice_response",
                        "response": response.model_dump(),
                        "latency_metrics": metrics.to_dict()
                    })
                    audio_buffer.clear()

            elif "text" in data:
                # Handle text query over websocket
                text_query = data["text"]
                response, metrics = await engine_instance.process_query(text_query)
                await websocket.send_json({
                    "event": "text_response",
                    "response": response.model_dump(),
                    "latency_metrics": metrics.to_dict()
                })

    except WebSocketDisconnect:
        logger.info("WebSocket voice stream connection closed.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
