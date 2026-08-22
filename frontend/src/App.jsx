"use client";

import React, { useState } from "react";
import HeroReveal from "./components/HeroReveal.jsx";
import VoiceRAG from "./components/VoiceRAG.jsx";
import {
  Mic,
  Sparkles,
  Zap,
  Shield,
  Layers,
  Terminal,
  Activity,
  Github,
  Sun,
  Flame,
  Radio,
  Server,
} from "lucide-react";

export default function App() {
  const [pipelineState, setPipelineState] = useState("ready");

  const scrollToConsole = () => {
    const el = document.getElementById("voice-rag-console");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="app-container">
      {/* Noise Texture Overlay */}
      <div className="noise-overlay" />

      {/* Stitch Top Nav Bar */}
      <nav className="stitch-navbar">
        <div className="nav-left">
          <div className="nav-brand-logo">
            <Flame size={16} className="text-black" />
          </div>
          <span className="nav-brand-text">Ask Anything</span>
        </div>

        <div className="nav-links">
          <a href="#research" className="nav-link font-mono">Research</a>
          <a href="#docs" className="nav-link font-mono">Docs</a>
          <a href="#status" className="nav-link font-mono">Status</a>
          <a
            href="https://github.com/surf3rr/Vellrag"
            target="_blank"
            rel="noreferrer"
            className="nav-link font-mono"
          >
            GitHub
          </a>
        </div>

        <div className="nav-actions">
          <button className="btn-nav-primary font-mono" onClick={scrollToConsole}>
            Get Started
          </button>
        </div>
      </nav>

      {/* Main Page Layout */}
      <main className="app-main">
        {/* 1. Hero Reveal Section with WebGL2 Dawn Bloom */}
        <HeroReveal
          eyebrow="ASK ANYTHING"
          title="Voice-Enabled RAG"
          subtitle="Speak the question. Get a grounded, cited answer — end to end, under 200ms."
          onDawnComplete={() => setPipelineState("active")}
          onStartTalking={scrollToConsole}
        />

        {/* Feature Summary Strip */}
        <section className="stitch-feature-strip">
          <div className="feature-card">
            <div className="feature-icon coral">
              <Zap size={18} />
            </div>
            <div className="feature-content">
              <h4>Sub-200ms Target</h4>
              <p>Optimized pipeline with BGE embeddings, in-memory Qdrant HNSW, and Groq LLaMA 3.1 inference.</p>
            </div>
          </div>

          <div className="feature-card">
            <div className="feature-icon amber">
              <Layers size={18} />
            </div>
            <div className="feature-content">
              <h4>Hierarchical Chunking</h4>
              <p>128-token child chunks for dense vector search mapped to 512-token parent chunks for LLM context.</p>
            </div>
          </div>

          <div className="feature-card">
            <div className="feature-icon edge">
              <Shield size={18} />
            </div>
            <div className="feature-content">
              <h4>Pydantic V2 Guardrails</h4>
              <p>Pre-query safety and prompt injection defense with factual grounding thresholding (&gt;0.60).</p>
            </div>
          </div>
        </section>

        {/* 2. Voice RAG 4-Panel Console */}
        <VoiceRAG />
      </main>

      {/* Editorial Footer */}
      <footer className="stitch-footer">
        <div className="footer-left font-mono">
          © 2026 Ask Anything. Built in the Hacker House Goa.
        </div>
        <div className="footer-right font-mono">
          <span>P50: 86.1ms</span>
          <span>•</span>
          <span>Qdrant HNSW</span>
          <span>•</span>
          <span>Groq 8B Instant</span>
        </div>
      </footer>

      {/* Global Component Styles */}
      <style>{`
        .app-container {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          padding: 0 1.5rem;
          position: relative;
        }

        .stitch-navbar {
          max-width: 1200px;
          width: 100%;
          margin: 1.25rem auto 2rem auto;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.85rem 1.5rem;
          background: rgba(21, 19, 16, 0.7);
          border: 1px solid var(--line);
          border-radius: var(--radius-full);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          z-index: 50;
        }

        .nav-left {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .nav-brand-logo {
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: var(--primary-container);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .nav-brand-text {
          font-family: var(--font-display);
          font-weight: 700;
          font-size: 1.05rem;
          color: var(--primary);
        }

        .nav-links {
          display: flex;
          align-items: center;
          gap: 1.75rem;
        }

        .nav-link {
          font-size: 0.82rem;
          color: var(--on-surface-variant);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          transition: color 0.2s ease;
        }

        .nav-link:hover {
          color: var(--primary);
        }

        .btn-nav-primary {
          background: var(--primary-container);
          color: #ffffff;
          font-size: 0.78rem;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          padding: 0.5rem 1.2rem;
          border-radius: var(--radius-full);
          font-weight: 600;
          transition: all 0.2s ease;
          box-shadow: 0 0 14px rgba(255, 90, 60, 0.35);
        }

        .btn-nav-primary:hover {
          background: #ff7056;
          transform: translateY(-1px);
        }

        .app-main {
          flex: 1;
          display: flex;
          flex-direction: column;
        }

        .stitch-feature-strip {
          max-width: 1200px;
          width: 100%;
          margin: 0 auto 3rem auto;
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1.25rem;
        }

        .feature-card {
          background: var(--surface-container);
          border: 1px solid var(--line);
          border-radius: var(--radius-lg);
          padding: 1.35rem;
          display: flex;
          align-items: flex-start;
          gap: 1rem;
          backdrop-filter: blur(12px);
        }

        .feature-icon {
          width: 38px;
          height: 38px;
          border-radius: var(--radius-sm);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .feature-icon.coral {
          background: rgba(255, 90, 60, 0.12);
          color: var(--primary-container);
        }

        .feature-icon.amber {
          background: rgba(255, 185, 82, 0.12);
          color: var(--secondary);
        }

        .feature-icon.edge {
          background: rgba(255, 46, 99, 0.12);
          color: var(--ink-edge);
        }

        .feature-content h4 {
          font-size: 0.95rem;
          color: var(--on-surface);
          margin-bottom: 0.3rem;
        }

        .feature-content p {
          font-size: 0.8rem;
          line-height: 1.5;
          color: var(--on-surface-variant);
        }

        .stitch-footer {
          max-width: 1200px;
          width: 100%;
          margin: 0 auto;
          padding: 2.25rem 0 3rem 0;
          border-top: 1px solid var(--line);
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 1rem;
          font-size: 0.78rem;
          color: var(--on-surface-variant);
          opacity: 0.8;
        }

        .footer-right {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: var(--secondary);
        }

        @media (max-width: 860px) {
          .nav-links { display: none; }
          .stitch-feature-strip { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  );
}
