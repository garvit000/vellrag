"""
Guardrails and Validation Module using Pydantic V2.
Includes safety verification, prompt injection detection, and context grounding checks.
"""

import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class GuardrailResponse(BaseModel):
    """
    Structured Output Schema enforcing strict Voice RAG Operational Rules.
    """
    is_safe: bool = Field(description="True if query is safe and topic is valid.")
    context_grounded: bool = Field(description="True if retrieved context has sufficient facts to answer.")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")
    refusal_reason: Optional[str] = Field(default=None, description="Brief reason if is_safe or context_grounded is false.")
    answer: str = Field(description="Concise voice-friendly answer (no Markdown, tables, bullets) or polite fallback.")


# Known prompt injection & jailbreak patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+)?instructions",
    r"reveal\s+(backend\s+)?system\s+prompt",
    r"you\s+are\s+now\s+in\s+dan\s+mode",
    r"system\s+override",
    r"disregard\s+operational\s+rules",
    r"show\s+me\s+your\s+prompt\s+template",
]

# Off-topic / out-of-scope keywords for basic filter
UNSAFE_CONTENT_PATTERNS = [
    r"how\s+to\s+(make|build)\s+(a\s+)?bomb",
    r"malware\s+code",
    r"illegal\s+activities",
]


class GuardrailEngine:
    """Engine executing pre-query and post-retrieval safety and grounding checks."""

    def __init__(
        self,
        similarity_threshold: float = settings.GROUNDING_SIMILARITY_THRESHOLD
    ):
        self.similarity_threshold = similarity_threshold

    def validate_query_safety(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Pre-execution check for prompt injection, jailbreaks, and toxic requests.
        Returns (is_safe, refusal_reason).
        """
        query_lower = query.strip().lower()

        # Check prompt injection
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"Guardrail Flagged Prompt Injection: {query}")
                return False, "I cannot fulfill requests that attempt to bypass safety guidelines or reveal system prompts."

        # Check unsafe content
        for pattern in UNSAFE_CONTENT_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"Guardrail Flagged Unsafe Content: {query}")
                return False, "I am unable to assist with unsafe or restricted queries."

        return True, None

    def evaluate_grounding(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Post-retrieval factual grounding evaluation.
        Checks maximum similarity score among retrieved chunks against threshold.
        Returns (context_grounded, confidence_score, refusal_reason).
        """
        if not retrieved_contexts:
            return False, 0.0, "I don't have enough grounded context in my database to answer that."

        max_score = max((hit.get("score", 0.0) for hit in retrieved_contexts), default=0.0)

        if max_score < self.similarity_threshold:
            logger.info(f"Retrieval score {max_score:.4f} below threshold {self.similarity_threshold}.")
            return False, round(max_score, 4), "I don't have enough grounded context in my database to answer that."

        confidence = min(1.0, max(0.5, max_score))
        return True, round(confidence, 4), None

    def format_voice_answer(self, raw_text: str) -> str:
        """
        Sanitize raw LLM response for Text-to-Speech (TTS):
        Removes Markdown headers, tables, asterisks, bullet points, and LaTeX symbols.
        Ensures concise 2-3 sentence limit.
        """
        # Remove markdown formatting (*, #, `, _, ~, [], ())
        text = re.sub(r'[\*\#\`\_\~\[\]]', '', raw_text)
        text = re.sub(r'\$(\$?)(.*?)\1\$', r'\2', text)  # remove latex math
        text = re.sub(r'^\s*[-+*]\s+', '', text, flags=re.MULTILINE)  # remove bullet points
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # remove numbered lists
        text = re.sub(r'\n+', ' ', text).strip()  # flatten newlines

        # Limit to 2-3 sentences for voice conciseness
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 3:
            text = " ".join(sentences[:3])

        return text
