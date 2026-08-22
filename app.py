"""
FastAPI Web Server providing REST and WebSocket Endpoints for Voice RAG System.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
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

# Global Engine Instance with dynamic STT Client binding
stt_client_instance = None
if settings.SARVAM_API_KEY and not ("your_sarvam" in settings.SARVAM_API_KEY or settings.SARVAM_API_KEY.startswith("mock")):
    stt_client_instance = SarvamSTTClient(api_key=settings.SARVAM_API_KEY)
    logger.info("Initialized Sarvam AI STT Client.")
elif settings.ELEVENLABS_API_KEY and not ("your_elevenlabs" in settings.ELEVENLABS_API_KEY or settings.ELEVENLABS_API_KEY.startswith("mock")):
    stt_client_instance = ElevenLabsSTTClient(api_key=settings.ELEVENLABS_API_KEY)
    logger.info("Initialized ElevenLabs STT Client.")
else:
    stt_client_instance = MockStreamingSTTClient()
    logger.info("Initialized Mock Streaming STT Client.")

indexer_instance = VectorIndexer()
engine_instance = VoiceRAGEngine(indexer=indexer_instance, stt_client=stt_client_instance)


class TextQueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is the target latency for voice RAG pipelines?"})
    top_k: int = Field(default=2, ge=1, le=5)


class IndexRequest(BaseModel):
    num_documents: int = Field(default=5, ge=1, le=50)
    strategy: str = Field(default="hierarchical", json_schema_extra={"example": "hierarchical"})


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
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice RAG Control Dashboard</title>
    <meta name="description" content="Low-latency Voice-Enabled Retrieval-Augmented Generation (RAG) Control Center. Monitor latency percentiles, trigger indexing, and query the vector store via text or voice.">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
            background-attachment: fixed;
        }

        header {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .header-title-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        h1 {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #a5b4fc, #6366f1, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            padding: 4px 10px;
            border-radius: 9999px;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            display: inline-block;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        main {
            flex: 1;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            padding: 32px;
            box-sizing: border-box;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 28px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
        }

        h2 {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 4px 0;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-desc {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin: 0 0 16px 0;
        }

        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 16px;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 10px 16px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }

        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-row {
            display: flex;
            gap: 12px;
        }

        input[type="text"], input[type="number"] {
            flex: 1;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px 16px;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.95rem;
            transition: all 0.2s;
        }

        input[type="text"]:focus, input[type="number"]:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        button.btn-primary {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
            justify-content: center;
        }

        button.btn-primary:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
        }

        button.btn-primary:active {
            transform: translateY(0);
        }

        button.btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        button.btn-secondary:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .mic-btn-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
            padding: 24px;
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.4);
        }

        .mic-btn {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: rgba(99, 102, 241, 0.1);
            border: 2px solid var(--primary);
            color: var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1.8rem;
        }

        .mic-btn.recording {
            background: rgba(239, 68, 68, 0.1);
            border-color: var(--error);
            color: var(--error);
            animation: pulse-recording 1.5s infinite;
        }

        @keyframes pulse-recording {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        .latency-container {
            display: flex;
            flex-direction: column;
            gap: 16px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }

        .latency-title-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
        }

        .total-time-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--primary);
            background: rgba(99, 102, 241, 0.1);
            padding: 4px 12px;
            border-radius: 8px;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }

        .total-time-badge.compliant {
            color: var(--success);
            background: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.2);
        }

        .total-time-badge.non-compliant {
            color: var(--warning);
            background: rgba(245, 158, 11, 0.1);
            border-color: rgba(245, 158, 11, 0.2);
        }

        .bar-chart {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .bar-row {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .bar-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .bar-bg {
            background: rgba(255, 255, 255, 0.05);
            height: 10px;
            border-radius: 9999px;
            overflow: hidden;
            position: relative;
        }

        .bar-fill {
            height: 100%;
            border-radius: 9999px;
            width: 0%;
            transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .stt-fill { background: linear-gradient(90deg, #818cf8, #6366f1); }
        .emb-fill { background: linear-gradient(90deg, #c084fc, #a855f7); }
        .ret-fill { background: linear-gradient(90deg, #60a5fa, #3b82f6); }
        .llm-fill { background: linear-gradient(90deg, #34d399, #10b981); }

        .terminal-output {
            background: #060913;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            padding: 16px;
            height: 250px;
            overflow-y: auto;
            color: #e2e8f0;
            display: flex;
            flex-direction: column;
            gap: 8px;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
        }

        .terminal-line {
            line-height: 1.4;
        }

        .terminal-info { color: #38bdf8; }
        .terminal-success { color: #4ade80; }
        .terminal-warning { color: #fbbf24; }
        .terminal-error { color: #f87171; }

        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        .meta-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .meta-val {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .table-container {
            overflow-x: auto;
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }

        th {
            background: rgba(15, 23, 42, 0.8);
            padding: 12px 16px;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
        }

        td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-main);
        }

        tr:last-child td {
            border-bottom: none;
        }

        .confidence-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .confidence-high { background: rgba(16, 185, 129, 0.1); color: var(--success); }
        .confidence-medium { background: rgba(245, 158, 11, 0.1); color: var(--warning); }
        .confidence-low { background: rgba(239, 68, 68, 0.1); color: var(--error); }

        footer {
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border-color);
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title-container">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="#6366f1"/>
            </svg>
            <h1 id="title-main">Voice RAG Control Dashboard</h1>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            <span id="txt-status">System Active</span>
        </div>
    </header>

    <main>
        <!-- Left Panel: Interaction & Controls -->
        <div class="panel">
            <div>
                <h2>RAG Query Center</h2>
                <div class="section-desc">Interact with the Voice RAG system via text or real-time microphone stream.</div>
            </div>

            <div class="tabs">
                <button class="tab-btn active" id="tab-text" onclick="switchTab('text')">Text Input</button>
                <button class="tab-btn" id="tab-voice" onclick="switchTab('voice')">Voice Stream</button>
            </div>

            <div id="content-text" class="tab-content active">
                <div class="form-group">
                    <label for="text-query-input">Ask RAG System</label>
                    <div class="input-row">
                        <input type="text" id="text-query-input" placeholder="e.g. What is the target latency for voice RAG pipelines?" onkeydown="if(event.key === 'Enter') sendTextQuery()">
                        <button class="btn-primary" id="btn-submit-query" onclick="sendTextQuery()">
                            Submit
                        </button>
                    </div>
                </div>
            </div>

            <div id="content-voice" class="tab-content">
                <div class="mic-btn-container">
                    <button class="mic-btn" id="btn-start-voice" onclick="toggleVoiceStreaming()">
                        🎤
                    </button>
                    <div id="voice-stream-label" style="font-weight: 500; font-size: 0.9rem;">Click to start streaming microphone audio</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-align: center;">Captures downsampled 16kHz PCM chunks and streams via WebSockets to `/ws/voice`</div>
                </div>
            </div>

            <div class="latency-container">
                <div class="latency-title-row">
                    <h3 style="margin: 0; font-size: 0.95rem; font-weight: 600;">Last Query Latency Breakdown</h3>
                    <div class="total-time-badge" id="lbl-total-latency">0.00 ms</div>
                </div>

                <div class="bar-chart">
                    <div class="bar-row">
                        <div class="bar-label-row">
                            <span>1. Speech-to-Text (STT)</span>
                            <span id="lbl-stt-ms">0.00 ms</span>
                        </div>
                        <div class="bar-bg">
                            <div id="bar-stt" class="bar-fill stt-fill"></div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label-row">
                            <span>2. Query Embedding</span>
                            <span id="lbl-emb-ms">0.00 ms</span>
                        </div>
                        <div class="bar-bg">
                            <div id="bar-emb" class="bar-fill emb-fill"></div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label-row">
                            <span>3. Vector Retrieval</span>
                            <span id="lbl-ret-ms">0.00 ms</span>
                        </div>
                        <div class="bar-bg">
                            <div id="bar-ret" class="bar-fill ret-fill"></div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label-row">
                            <span>4. LLM Generation</span>
                            <span id="lbl-llm-ms">0.00 ms</span>
                        </div>
                        <div class="bar-bg">
                            <div id="bar-llm" class="bar-fill llm-fill"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div>
                <h2>System Configurations</h2>
                <div class="section-desc">Active parameters and environment bindings fetched from `config.py`.</div>
                <div class="metadata-grid">
                    <div class="meta-card">
                        <label>Embedding Model</label>
                        <div class="meta-val" style="font-size: 0.85rem;" id="lbl-meta-emb">BAAI/bge-small-en-v1.5</div>
                    </div>
                    <div class="meta-card">
                        <label>Vector DB Collection</label>
                        <div class="meta-val" id="lbl-meta-collection">msmarco_xl_chunks</div>
                    </div>
                    <div class="meta-card">
                        <label>Target Latency</label>
                        <div class="meta-val" id="lbl-meta-target">200.0 ms</div>
                    </div>
                    <div class="meta-card">
                        <label>Indexing strategy</label>
                        <div class="meta-val">Hierarchical</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Panel: Logs, Benchmarks, Indexing -->
        <div class="panel">
            <div>
                <h2>Real-Time Response Monitor</h2>
                <div class="section-desc">Answers, grounding checks, and raw WebSocket/REST communication outputs.</div>
            </div>

            <div class="terminal-output" id="terminal-log">
                <div class="terminal-line terminal-info">[System] Ready. Awaiting user interaction...</div>
            </div>

            <div>
                <h2>Indexing Control</h2>
                <div class="section-desc">Rebuild the vector store using MSMARCO-Xl documents.</div>
                <div class="form-group">
                    <div class="input-row">
                        <input type="number" id="input-index-docs" value="5" min="1" max="50" style="max-width: 100px;">
                        <button class="btn-secondary" id="btn-index-docs" onclick="triggerIndexing()">Index Documents</button>
                    </div>
                </div>
            </div>

            <div>
                <h2>Benchmarking Suite</h2>
                <div class="section-desc">Trigger standard latency benchmarking over 10 check queries.</div>
                <button class="btn-primary" id="btn-run-benchmark" onclick="runBenchmark()">Run Latency Benchmark</button>
                
                <div id="benchmark-results-container" style="display: none; margin-top: 16px; flex-direction: column; gap: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.9rem;">Benchmark Output</span>
                        <span class="confidence-indicator confidence-high" id="lbl-benchmark-compliance">100% Compliant</span>
                    </div>
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Phase</th>
                                    <th>P50 (ms)</th>
                                    <th>P70 (ms)</th>
                                    <th>P100 (ms)</th>
                                </tr>
                            </thead>
                            <tbody id="benchmark-table-body">
                                <!-- Populated dynamically -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        Voice RAG System (HH Goa 2026 Task 2) | Ultra-Low Latency Conversational Q&A
    </footer>

    <script>
        let wsStream;
        let audioContext;
        let processor;
        let mediaStream;
        let isRecording = false;

        // Fetch startup configs
        async function fetchConfig() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('lbl-meta-emb').textContent = data.embedding_model;
                document.getElementById('lbl-meta-collection').textContent = data.collection;
                document.getElementById('lbl-meta-target').textContent = data.target_latency_ms + " ms";
            } catch (err) {
                console.error("Failed to load configs", err);
            }
        }
        window.addEventListener('load', fetchConfig);

        function logToTerminal(message, type = 'info') {
            const term = document.getElementById('terminal-log');
            const line = document.createElement('div');
            line.className = `terminal-line terminal-${type}`;
            const time = new Date().toLocaleTimeString();
            line.innerHTML = `[${time}] ${message}`;
            term.appendChild(line);
            term.scrollTop = term.scrollHeight;
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            document.getElementById(`tab-${tab}`).classList.add('active');
            document.getElementById(`content-${tab}`).classList.add('active');
            
            logToTerminal(`Switched interaction mode to: ${tab.toUpperCase()}`);
            if (isRecording) {
                stopVoiceStreaming();
            }
        }

        async function sendTextQuery() {
            const input = document.getElementById('text-query-input');
            const query = input.value.trim();
            if (!query) return;

            input.value = '';
            logToTerminal(`Sending text query: "${query}"`, 'info');

            try {
                const response = await fetch('/api/v1/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: 2 })
                });

                if (!response.ok) {
                    throw new Error(`HTTP Error ${response.status}`);
                }

                const data = await response.json();
                displayQueryResult(data);
            } catch (err) {
                logToTerminal(`Query failed: ${err.message}`, 'error');
            }
        }

        function displayQueryResult(data) {
            const answer = data.response.answer;
            const isSafe = data.response.is_safe;
            const grounded = data.response.context_grounded;
            const confidence = data.response.confidence_score;
            const metrics = data.latency_metrics;

            if (!isSafe) {
                logToTerminal(`Guardrail Warning: Safety filter flagged this request. Refusal reason: ${data.response.refusal_reason}`, 'warning');
            } else if (!grounded) {
                logToTerminal(`RAG Grounding Refusal: ${data.response.refusal_reason}`, 'warning');
            } else {
                logToTerminal(`RAG Output: ${answer}`, 'success');
                logToTerminal(`Grounding confidence score: ${confidence.toFixed(2)}`, 'info');
            }

            updateLatencyTimeline(metrics);
        }

        function updateLatencyTimeline(metrics) {
            const total = metrics.total_latency_ms;
            document.getElementById('lbl-total-latency').textContent = `${total.toFixed(2)} ms`;
            
            const badge = document.getElementById('lbl-total-latency');
            badge.className = 'total-time-badge';
            if (total <= 200.0) {
                badge.classList.add('compliant');
            } else {
                badge.classList.add('non-compliant');
            }

            // Update percentage bars
            const setBarWidth = (barId, lblId, val) => {
                const percentage = Math.min(100, (val / Math.max(1, total)) * 100);
                document.getElementById(barId).style.width = `${percentage}%`;
                document.getElementById(lblId).textContent = `${val.toFixed(2)} ms`;
            };

            setBarWidth('bar-stt', 'lbl-stt-ms', metrics.stt_latency_ms);
            setBarWidth('bar-emb', 'lbl-emb-ms', metrics.embedding_latency_ms);
            setBarWidth('bar-ret', 'lbl-ret-ms', metrics.retrieval_latency_ms);
            setBarWidth('bar-llm', 'lbl-llm-ms', metrics.llm_latency_ms);
        }

        async function triggerIndexing() {
            const numDocs = parseInt(document.getElementById('input-index-docs').value) || 5;
            logToTerminal(`Triggering indexing of ${numDocs} MSMARCO documents into memory vector DB...`, 'info');

            try {
                const res = await fetch('/api/v1/index', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ num_documents: numDocs, strategy: 'hierarchical' })
                });
                const data = await res.json();
                logToTerminal(`Indexing Complete: ${data.message} Indexed ${data.chunks_indexed} hierarchical child chunks.`, 'success');
            } catch (err) {
                logToTerminal(`Indexing failed: ${err.message}`, 'error');
            }
        }

        async function runBenchmark() {
            logToTerminal("Executing sub-200ms latency benchmark suite (10 queries, 1 iteration)...", 'info');
            const btn = document.getElementById('btn-run-benchmark');
            btn.disabled = true;
            btn.textContent = "Benchmarking...";

            try {
                const res = await fetch('/api/v1/benchmark?iterations=1');
                const data = await res.json();
                
                logToTerminal(`Benchmark completed successfully! Compliance rate: ${data.sub_200ms_compliance_rate}%`, 'success');
                
                document.getElementById('benchmark-results-container').style.display = 'flex';
                document.getElementById('lbl-benchmark-compliance').textContent = `${data.sub_200ms_compliance_rate}% Compliant`;
                
                const tableBody = document.getElementById('benchmark-table-body');
                tableBody.innerHTML = '';

                const addRow = (phase, m) => {
                    const row = `<tr>
                        <td><strong>${phase}</strong></td>
                        <td>${m.p50.toFixed(2)} ms</td>
                        <td>${m.p70.toFixed(2)} ms</td>
                        <td>${m.p100.toFixed(2)} ms</td>
                    </tr>`;
                    tableBody.innerHTML += row;
                };

                addRow('1. Speech-to-Text (STT)', data.stt_metrics);
                addRow('2. Query Embedding', data.embedding_metrics);
                addRow('3. Vector Retrieval', data.retrieval_metrics);
                addRow('4. LLM Generation', data.llm_metrics);
                addRow('TOTAL PIPELINE', data.total_metrics);

            } catch (err) {
                logToTerminal(`Benchmark failed: ${err.message}`, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = "Run Latency Benchmark";
            }
        }

        async function toggleVoiceStreaming() {
            if (isRecording) {
                await stopVoiceStreaming();
            } else {
                await startVoiceStreaming();
            }
        }

        let recognitionInstance = null;

        async function startVoiceStreaming() {
            logToTerminal("Initializing Voice Microphone Stream...", 'info');
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

            if (SpeechRecognition) {
                if (!recognitionInstance) {
                    recognitionInstance = new SpeechRecognition();
                    recognitionInstance.continuous = true;
                    recognitionInstance.interimResults = true;
                    recognitionInstance.lang = 'en-US';

                    recognitionInstance.onstart = () => {
                        isRecording = true;
                        document.getElementById('btn-start-voice').classList.add('recording');
                        document.getElementById('voice-stream-label').textContent = "Listening... Speak into your microphone!";
                        logToTerminal("Microphone active! Listening to your voice...", 'success');
                    };

                    recognitionInstance.onresult = async (event) => {
                        let finalTranscript = '';
                        let interimTranscript = '';

                        for (let i = event.resultIndex; i < event.results.length; ++i) {
                            if (event.results[i].isFinal) {
                                finalTranscript += event.results[i][0].transcript;
                            } else {
                                interimTranscript += event.results[i][0].transcript;
                            }
                        }

                        if (interimTranscript) {
                            document.getElementById('voice-stream-label').textContent = `Hearing: "${interimTranscript}"`;
                        }

                        if (finalTranscript) {
                            logToTerminal(`Live Voice Transcribed: "${finalTranscript.trim()}"`, 'success');
                            document.getElementById('voice-stream-label').textContent = `Transcribed: "${finalTranscript.trim()}"`;
                            document.getElementById('text-query-input').value = finalTranscript.trim();
                            await sendTextQuery();
                        }
                    };

                    recognitionInstance.onerror = (event) => {
                        logToTerminal(`Speech Recognition event: ${event.error}`, 'info');
                    };

                    recognitionInstance.onend = () => {
                        if (isRecording) {
                            try { recognitionInstance.start(); } catch(e) {}
                        }
                    };
                }

                try {
                    recognitionInstance.start();
                } catch (err) {
                    logToTerminal(`Could not start speech recognition: ${err.message}`, 'error');
                }
            } else {
                await startWebSocketAudioStreaming();
            }
        }

        async function startWebSocketAudioStreaming() {
            logToTerminal("Establishing WebSocket voice connection...", 'info');
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            wsStream = new WebSocket(`${protocol}//${window.location.host}/ws/voice`);
            wsStream.binaryType = 'arraybuffer';

            wsStream.onopen = async () => {
                logToTerminal("WebSocket connection open. Capturing microphone audio...", 'info');
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    
                    processor = audioContext.createScriptProcessor(4096, 1, 1);
                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    processor.onaudioprocess = (e) => {
                        if (wsStream && wsStream.readyState === WebSocket.OPEN) {
                            const inputData = e.inputBuffer.getChannelData(0);
                            const pcmBuffer = new Int16Array(inputData.length);
                            for (let i = 0; i < inputData.length; i++) {
                                pcmBuffer[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                            }
                            wsStream.send(pcmBuffer.buffer);
                        }
                    };

                    isRecording = true;
                    document.getElementById('btn-start-voice').classList.add('recording');
                    document.getElementById('voice-stream-label').textContent = "Streaming microphone... Click button to stop.";
                    logToTerminal("Streaming PCM audio chunks in real-time.", 'success');
                } catch (err) {
                    logToTerminal(`Microphone access error: ${err.message}`, 'error');
                    wsStream.close();
                }
            };

            wsStream.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.event === 'voice_response') {
                    logToTerminal(`WebSocket audio chunk processed. Query transcribed as: "${data.response.answer}"`, 'success');
                    displayQueryResult({ response: data.response, latency_metrics: data.latency_metrics });
                } else if (data.event === 'text_response') {
                    displayQueryResult({ response: data.response, latency_metrics: data.latency_metrics });
                }
            };

            wsStream.onerror = (e) => {
                logToTerminal("WebSocket error occurred.", 'error');
            };

            wsStream.onclose = () => {
                isRecording = false;
                document.getElementById('btn-start-voice').classList.remove('recording');
                document.getElementById('voice-stream-label').textContent = "Click to start streaming microphone audio";
                logToTerminal("WebSocket connection closed.", 'info');
            };
        }

        async function stopVoiceStreaming() {
            logToTerminal("Stopping microphone capture...", 'info');
            isRecording = false;
            if (recognitionInstance) {
                try { recognitionInstance.stop(); } catch(e) {}
            }
            if (processor) {
                processor.disconnect();
                processor = null;
            }
            if (audioContext) {
                await audioContext.close();
                audioContext = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            if (wsStream) {
                wsStream.close();
                wsStream = null;
            }
            document.getElementById('btn-start-voice').classList.remove('recording');
            document.getElementById('voice-stream-label').textContent = "Click to start streaming microphone audio";
        }
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTML_CONTENT


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


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
    docs = load_msmarco_xl_dataset(limit=req.num_documents, fetch_remote=True)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=False)
