"""
Async STT Adapter Module supporting Sarvam AI, ElevenLabs, and Mock Streaming STT.
"""

import abc
import asyncio
import time
import logging
from typing import AsyncGenerator, Optional, Dict, Any
import httpx

from config import settings

logger = logging.getLogger(__name__)


class STTResult:
    def __init__(self, text: str, latency_ms: float, is_final: bool = True, confidence: float = 0.95):
        self.text = text
        self.latency_ms = latency_ms
        self.is_final = is_final
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "latency_ms": self.latency_ms,
            "is_final": self.is_final,
            "confidence": self.confidence
        }


class BaseSTTClient(abc.ABC):
    """Abstract Base Class for Speech-to-Text Async Adapters."""

    @abc.abstractmethod
    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> STTResult:
        """Transcribe a chunk or complete audio buffer."""
        pass

    @abc.abstractmethod
    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[STTResult, None]:
        """Process streaming audio chunks asynchronously."""
        pass


class GroqWhisperSTTClient(BaseSTTClient):
    """
    Groq Whisper Turbo Async STT Adapter.
    Endpoint: https://api.groq.com/openai/v1/audio/transcriptions
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-large-v3-turbo"):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model
        self.endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "speech.webm", sample_rate: int = 16000) -> STTResult:
        start_time = time.perf_counter()
        if not self.api_key or self.api_key.startswith("mock"):
            await asyncio.sleep(0.025)
            return STTResult(
                text="What is the target latency for voice RAG systems?",
                latency_ms=(time.perf_counter() - start_time) * 1000.0
            )

        headers = {"Authorization": f"Bearer {self.api_key}"}
        mime_type = "audio/webm" if filename.endswith(".webm") else "audio/wav"
        files = {"file": (filename, audio_bytes, mime_type)}
        data = {"model": self.model, "response_format": "json", "temperature": 0.0}

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(self.endpoint, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    res_json = response.json()
                    transcript = res_json.get("text", "").strip()
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    return STTResult(text=transcript, latency_ms=latency_ms)
                else:
                    logger.warning(f"Groq Whisper returned HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Groq Whisper transcription exception: {e}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return STTResult(text="", latency_ms=latency_ms)

    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[STTResult, None]:
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)
            if len(buffer) >= 16000 * 2:
                result = await self.transcribe_bytes(bytes(buffer))
                yield result
                buffer.clear()
        if buffer:
            yield await self.transcribe_bytes(bytes(buffer))


class SarvamSTTClient(BaseSTTClient):
    """
    Sarvam AI Async STT Adapter.
    Endpoint: https://api.sarvam.ai/speech-to-text
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.endpoint = "https://api.sarvam.ai/speech-to-text"

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "speech.wav", sample_rate: int = 16000) -> STTResult:
        start_time = time.perf_counter()
        if not self.api_key or self.api_key.startswith("mock"):
            await asyncio.sleep(0.035)
            return STTResult(
                text="What is the target latency for voice RAG pipelines?",
                latency_ms=(time.perf_counter() - start_time) * 1000.0
            )

        headers = {"api-subscription-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": "saarika:v2.5", "language_code": "en-IN"}

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(self.endpoint, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    res_data = response.json()
                    transcript = res_data.get("transcript", "").strip()
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    return STTResult(text=transcript, latency_ms=latency_ms)
                else:
                    logger.warning(f"Sarvam AI returned HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Sarvam AI transcription exception: {e}")
            
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return STTResult(text="", latency_ms=latency_ms)

    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[STTResult, None]:
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)
            if len(buffer) >= 16000 * 2:
                result = await self.transcribe_bytes(bytes(buffer))
                yield result
                buffer.clear()
        if buffer:
            yield await self.transcribe_bytes(bytes(buffer))


class ElevenLabsSTTClient(BaseSTTClient):
    """
    ElevenLabs STT Async Adapter.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ELEVENLABS_API_KEY
        self.endpoint = "https://api.elevenlabs.io/v1/speech-to-text"

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "speech.wav", sample_rate: int = 16000) -> STTResult:
        start_time = time.perf_counter()
        if not self.api_key or self.api_key.startswith("mock"):
            await asyncio.sleep(0.040)
            return STTResult(
                text="How does parent document chunking work?",
                latency_ms=(time.perf_counter() - start_time) * 1000.0
            )

        headers = {"xi-api-key": self.api_key}
        files = {"file": (filename, audio_bytes, "audio/wav")}

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(self.endpoint, headers=headers, files=files)
                if response.status_code == 200:
                    res_data = response.json()
                    transcript = res_data.get("text", "").strip()
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    return STTResult(text=transcript, latency_ms=latency_ms)
                else:
                    logger.warning(f"ElevenLabs returned HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"ElevenLabs transcription exception: {e}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return STTResult(text="", latency_ms=latency_ms)

    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[STTResult, None]:
        buffer = bytearray()
        async for chunk in audio_stream:
            buffer.extend(chunk)
            if len(buffer) >= 16000:
                result = await self.transcribe_bytes(bytes(buffer))
                yield result
                buffer.clear()
        if buffer:
            yield await self.transcribe_bytes(bytes(buffer))


class CompositeSTTClient(BaseSTTClient):
    """
    Composite STT Adapter that cascades across Groq Whisper, Sarvam AI, and ElevenLabs.
    """

    def __init__(self, clients: Optional[list] = None):
        self.clients = clients or []

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "speech.webm", sample_rate: int = 16000) -> STTResult:
        for client in self.clients:
            try:
                res = await client.transcribe_bytes(audio_bytes, filename=filename, sample_rate=sample_rate)
                if res.text and len(res.text.strip()) > 0:
                    return res
            except Exception as e:
                logger.warning(f"STT client {client.__class__.__name__} failed: {e}")
                continue
        return STTResult(text="Could not transcribe audio.", latency_ms=10.0)

    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[STTResult, None]:
        if self.clients:
            async for res in self.clients[0].process_stream(audio_stream):
                yield res


class MockStreamingSTTClient(BaseSTTClient):
    """
    Ultra-low latency mock STT adapter for testing and benchmarking voice pipelines.
    Simulates live audio buffer transcription in ~10-25ms.
    """

    def __init__(self, default_text: Optional[str] = None, Simulated_latency_ms: float = 15.0):
        self.default_text = default_text
        self.simulated_latency_ms = Simulated_latency_ms

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "speech.wav", sample_rate: int = 16000) -> STTResult:
        start_time = time.perf_counter()
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        text = self.default_text or "What is the target latency for voice RAG pipelines?"
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return STTResult(text=text, latency_ms=latency_ms)

    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[STTResult, None]:
        chunk_count = 0
        async for chunk in audio_stream:
            chunk_count += 1
            start_time = time.perf_counter()
            await asyncio.sleep(0.005)
            yield STTResult(
                text=f"Streaming chunk {chunk_count}",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                is_final=False
            )
        yield await self.transcribe_bytes(b"final_buffer")
