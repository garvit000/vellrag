"""
Dataset Loader & Vector Indexing Engine using Qdrant and SentenceTransformers.
Target Dataset: ai4bharat/MSMARCO-Xl with local fallback support.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from config import settings
from chunking import (
    Chunk,
    ParentChildDoc,
    HierarchicalChunker,
    SemanticChunker,
    SlidingWindowChunker,
    estimate_token_count
)

logger = logging.getLogger(__name__)


# Sample MSMARCO-Xl documents for fallback / fast offline demonstration
SAMPLE_MSMARCO_DOCUMENTS = [
    {
        "doc_id": "msmarco_doc_001",
        "title": "Voice RAG Architectures and Low Latency Systems",
        "text": (
            "Voice RAG systems combine Speech-to-Text, Vector Retrieval, and Large Language Models "
            "to answer questions aloud in real time. Low latency is critical for conversational interfaces, "
            "where target pipeline response time is usually under 200 milliseconds. "
            "Key strategies to reduce latency include using fast embedding models like BGE-small, "
            "in-memory vector databases like Qdrant with HNSW cosine distance indexing, "
            "and stream-first processing for speech recognition."
        )
    },
    {
        "doc_id": "msmarco_doc_002",
        "title": "MSMARCO-Xl Multilingual Information Retrieval Dataset",
        "text": (
            "MSMARCO-Xl is a comprehensive multilingual dataset constructed by AI4Bharat for passage ranking "
            "and question answering tasks across Indian languages and English. "
            "It extends the classic MSMARCO passage retrieval benchmark into multilingual settings, "
            "allowing evaluation of cross-lingual search models and retrieval-augmented generation systems."
        )
    },
    {
        "doc_id": "msmarco_doc_003",
        "title": "Hierarchical Parent-Document Chunking in Vector Search",
        "text": (
            "Parent-document chunking resolves the trade-off between retrieval granularity and LLM context size. "
            "Small child chunks of approximately 128 tokens are indexed into vector stores to ensure high precision cosine similarity match. "
            "When a child chunk matches the query, the indexer retrieves its parent document of 512 tokens. "
            "This parent chunk provides sufficient surrounding context for the LLM without overwhelming the prompt token budget."
        )
    },
    {
        "doc_id": "msmarco_doc_004",
        "title": "Guardrails and Fact Grounding in RAG Systems",
        "text": (
            "To prevent hallucinations, production RAG systems enforce strict grounding checks. "
            "The system evaluates the cosine similarity score of retrieved passages against the user query. "
            "If the maximum similarity score falls below a grounding threshold such as 0.40, "
            "the guardrail flags the query as ungrounded and returns a polite fallback refusal: "
            "'I don't have enough grounded context in my database to answer that.'"
        )
    },
    {
        "doc_id": "msmarco_doc_005",
        "title": "Sarvam AI and ElevenLabs Audio Streaming",
        "text": (
            "Sarvam AI offers specialized low-latency Speech-to-Text models for Indic languages and Indian English accents. "
            "ElevenLabs provides real-time streaming audio interfaces for speech recognition and voice synthesis. "
            "Integrating STT adapters with asynchronous chunked audio buffers minimizes audio input latency."
        )
    }
]


class VectorIndexer:
    """Qdrant Vector DB Indexer and Retriever."""

    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
        embedding_model_name: str = settings.EMBEDDING_MODEL_NAME,
        qdrant_host: str = settings.QDRANT_HOST
    ):
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.client = QdrantClient(qdrant_host)
        
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()

        # In-memory store for parent chunks: {parent_id: parent_text}
        self.parent_store: Dict[str, str] = {}
        self._init_collection()

    def _init_collection(self):
        """Initialize or recreate Qdrant collection with HNSW Cosine distance."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_dim,
                distance=qmodels.Distance.COSINE
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100
            )
        )
        logger.info(f"Qdrant collection '{self.collection_name}' initialized with dim {self.vector_dim}.")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Compute sentence embeddings."""
        if not texts:
            return []
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def index_chunks(self, chunks: List[Chunk]):
        """Embed and upsert chunks into Qdrant."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.encode(texts)

        points = []
        for i, chunk in enumerate(chunks):
            payload = {
                "doc_id": chunk.metadata.doc_id,
                "chunk_index": chunk.metadata.chunk_index,
                "token_count": chunk.metadata.token_count,
                "strategy": chunk.metadata.strategy,
                "parent_id": chunk.metadata.parent_id,
                "text": chunk.text,
                **chunk.metadata.extra
            }
            points.append(
                qmodels.PointStruct(
                    id=i,  # or deterministic integer/UUID hash
                    vector=embeddings[i],
                    payload=payload
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Successfully indexed {len(points)} chunks into Qdrant.")

    def index_parent_child_docs(self, parent_child_docs: List[ParentChildDoc]):
        """Index child chunks into Qdrant and store parent chunks in parent_store."""
        all_child_chunks: List[Chunk] = []

        for item in parent_child_docs:
            self.parent_store[item.parent_chunk.id] = item.parent_chunk.text
            all_child_chunks.extend(item.child_chunks)

        self.index_chunks(all_child_chunks)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform vector search against Qdrant collection.
        If result is a child chunk with parent_id, injects parent document context.
        """
        start_time = time.perf_counter()
        query_vec = self.encode([query])[0]

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k
        )

        search_latency_ms = (time.perf_counter() - start_time) * 1000.0
        retrieved: List[Dict[str, Any]] = []

        for hit in results:
            payload = hit.payload or {}
            score = float(hit.score)
            parent_id = payload.get("parent_id")
            
            # Retrieve parent text if available, otherwise use chunk text
            if parent_id and parent_id in self.parent_store:
                context_text = self.parent_store[parent_id]
                is_parent_retrieved = True
            else:
                context_text = payload.get("text", "")
                is_parent_retrieved = False

            retrieved.append({
                "score": score,
                "chunk_text": payload.get("text", ""),
                "context_text": context_text,
                "doc_id": payload.get("doc_id", ""),
                "strategy": payload.get("strategy", ""),
                "parent_id": parent_id,
                "is_parent_retrieved": is_parent_retrieved,
                "retrieval_latency_ms": search_latency_ms
            })

        return retrieved


def load_msmarco_xl_dataset(limit: int = 10) -> List[Dict[str, str]]:
    """
    Attempts to load ai4bharat/MSMARCO-Xl from HuggingFace datasets library.
    Falls back gracefully to pre-bundled MSMARCO documents if network/dataset unavailable.
    """
    try:
        from datasets import load_dataset
        logger.info("Downloading/Loading ai4bharat/MSMARCO-Xl dataset sample...")
        ds = load_dataset("ai4bharat/MSMARCO-Xl", name="english", split="train", streaming=True)
        docs = []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            docs.append({
                "doc_id": f"msmarco_{i}",
                "title": item.get("title", f"Document {i}"),
                "text": item.get("passage", item.get("text", ""))
            })
        if docs:
            logger.info(f"Loaded {len(docs)} documents from ai4bharat/MSMARCO-Xl.")
            return docs
    except Exception as e:
        logger.warning(f"Could not load ai4bharat/MSMARCO-Xl via datasets ({e}). Utilizing sample documents.")
    
    return SAMPLE_MSMARCO_DOCUMENTS[:limit]
