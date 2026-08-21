"""
Multi-Strategy Chunking Module
Implements Semantic, Hierarchical (Parent-Child), and Sliding Window Chunkers.
"""

import re
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import numpy as np


class ChunkMetadata(BaseModel):
    doc_id: str
    chunk_index: int
    token_count: int
    strategy: str
    parent_id: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class ParentChildDoc(BaseModel):
    parent_chunk: Chunk
    child_chunks: List[Chunk]


def estimate_token_count(text: str) -> int:
    """Fast estimation of token count by whitespace/punctuation split."""
    return len(re.findall(r'\w+|[^\w\s]', text))


def split_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class SlidingWindowChunker:
    """Sliding Window Chunker with configurable overlap."""

    def __init__(self, window_size: int = 256, overlap: int = 64):
        self.window_size = window_size
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        words = text.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        step = max(1, self.window_size - self.overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            window_words = words[i : i + self.window_size]
            chunk_text = " ".join(window_words)
            token_count = estimate_token_count(chunk_text)

            meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=chunk_idx,
                token_count=token_count,
                strategy="sliding_window",
                extra={"start_word_idx": i, "end_word_idx": i + len(window_words)}
            )
            chunks.append(Chunk(text=chunk_text, metadata=meta))
            chunk_idx += 1

            if i + self.window_size >= len(words):
                break

        return chunks


class HierarchicalChunker:
    """
    Hierarchical (Parent-Child) Chunker.
    Generates small child chunks (~128 tokens) for fine-grained vector retrieval
    and maps them to parent chunks (~512 tokens) for LLM context injection.
    """

    def __init__(self, parent_size: int = 512, child_size: int = 128, overlap: int = 32):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str) -> List[ParentChildDoc]:
        words = text.split()
        if not words:
            return []

        parent_step = max(1, self.parent_size - self.overlap)
        result: List[ParentChildDoc] = []
        parent_idx = 0

        for i in range(0, len(words), parent_step):
            parent_words = words[i : i + self.parent_size]
            parent_text = " ".join(parent_words)
            parent_id = f"{doc_id}_parent_{parent_idx}"

            parent_meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=parent_idx,
                token_count=estimate_token_count(parent_text),
                strategy="hierarchical_parent",
                extra={"parent_id": parent_id}
            )
            parent_chunk = Chunk(id=parent_id, text=parent_text, metadata=parent_meta)

            # Generate child chunks for this parent
            child_chunks: List[Chunk] = []
            child_step = max(1, self.child_size - self.overlap)
            child_idx = 0

            for j in range(0, len(parent_words), child_step):
                child_words = parent_words[j : j + self.child_size]
                child_text = " ".join(child_words)
                child_meta = ChunkMetadata(
                    doc_id=doc_id,
                    chunk_index=child_idx,
                    token_count=estimate_token_count(child_text),
                    strategy="hierarchical_child",
                    parent_id=parent_id,
                    extra={"parent_id": parent_id}
                )
                child_chunks.append(Chunk(text=child_text, metadata=child_meta))
                child_idx += 1

                if j + self.child_size >= len(parent_words):
                    break

            result.append(ParentChildDoc(parent_chunk=parent_chunk, child_chunks=child_chunks))
            parent_idx += 1

            if i + self.parent_size >= len(words):
                break

        return result


class SemanticChunker:
    """
    Semantic Chunker.
    Splits text by sentence and groups consecutive sentences based on embedding similarity thresholds.
    """

    def __init__(self, similarity_threshold: float = 0.65, max_tokens: int = 512):
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens

    def chunk(self, doc_id: str, text: str, embedding_fn) -> List[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1:
            meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=0,
                token_count=estimate_token_count(sentences[0]),
                strategy="semantic"
            )
            return [Chunk(text=sentences[0], metadata=meta)]

        # Get sentence embeddings
        embeddings = embedding_fn(sentences)
        chunks: List[Chunk] = []
        current_sentences = [sentences[0]]
        current_token_count = estimate_token_count(sentences[0])
        chunk_idx = 0

        for i in range(1, len(sentences)):
            sent_text = sentences[i]
            sent_tokens = estimate_token_count(sent_text)

            # Cosine similarity between sentence i-1 and sentence i
            vec1 = np.array(embeddings[i - 1])
            vec2 = np.array(embeddings[i])
            similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9))

            if similarity >= self.similarity_threshold and (current_token_count + sent_tokens) <= self.max_tokens:
                current_sentences.append(sent_text)
                current_token_count += sent_tokens
            else:
                combined_text = " ".join(current_sentences)
                meta = ChunkMetadata(
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    token_count=current_token_count,
                    strategy="semantic",
                    extra={"sentence_count": len(current_sentences)}
                )
                chunks.append(Chunk(text=combined_text, metadata=meta))
                chunk_idx += 1
                current_sentences = [sent_text]
                current_token_count = sent_tokens

        if current_sentences:
            combined_text = " ".join(current_sentences)
            meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_index=chunk_idx,
                token_count=current_token_count,
                strategy="semantic",
                extra={"sentence_count": len(current_sentences)}
            )
            chunks.append(Chunk(text=combined_text, metadata=meta))

        return chunks
