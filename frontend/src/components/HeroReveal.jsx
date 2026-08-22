"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import "./HeroReveal.css";
import { Mic, RefreshCw, Sparkles, Zap, ShieldCheck } from "lucide-react";

/**
 * WebGL2 Vertex Shader (Full-screen quad)
 */
const VERTEX_SHADER_SRC = `#version 300 es
in vec2 a_position;
out vec2 v_uv;

void main() {
    v_uv = (a_position + 1.0) * 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

/**
 * WebGL2 Fragment Shader - Cinematic HH Goa Dawn Bloom
 * Colors from Stitch Solaris Pipeline:
 *  - Background Void: #0a0806 (vec3(0.039, 0.031, 0.024))
 *  - Sunrise Coral:   #ff5a3c (vec3(1.0, 0.353, 0.235))
 *  - Sunrise Amber:   #ffb952 (vec3(1.0, 0.725, 0.322))
 *  - Ink Edge:        #ff2e63 (vec3(1.0, 0.180, 0.388))
 *  - Panel Cream:     #f3ede1 (vec3(0.953, 0.929, 0.882))
 */
const FRAGMENT_SHADER_SRC = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_progress;   // 0.0 (night) to 1.0 (dawn settled)
uniform float u_time;       // continuous time in seconds
uniform vec2 u_resolution;  // canvas size in px

// 2D Hash & Noise Functions
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

// Fractional Brownian Motion for fluid ink bloom
float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.55;
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 4; ++i) {
        v += a * noise2D(p);
        p = rot * p * 2.05 + vec2(100.0);
        a *= 0.48;
    }
    return v;
}

void main() {
    vec2 st = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / min(u_resolution.x, u_resolution.y);
    float dist = length(st);
    float angle = atan(st.y, st.x);

    // Color Palette
    vec3 cNight = vec3(0.039, 0.031, 0.024);   // #0a0806
    vec3 cEdge  = vec3(1.0, 0.180, 0.388);    // #ff2e63
    vec3 cCoral = vec3(1.0, 0.353, 0.235);    // #ff5a3c
    vec3 cAmber = vec3(1.0, 0.725, 0.322);    // #ffb952
    vec3 cCream = vec3(0.953, 0.929, 0.882);   // #f3ede1

    // Organic wobble on the bloom boundary
    float wobble = fbm(vec2(cos(angle) * 2.2, sin(angle) * 2.2) + vec2(u_time * 0.1, u_progress * 0.3)) * 0.24;

    float p = smoothstep(0.0, 1.0, u_progress);
    float radius = p * 2.2;

    // Ink boundary mask
    float distortedDist = dist + wobble;
    float inkMask = 1.0 - smoothstep(radius - 0.25, radius + 0.05, distortedDist);
    float edgeGlow = smoothstep(radius - 0.22, radius, distortedDist) * (1.0 - smoothstep(radius, radius + 0.04, distortedDist));

    // Internal sunset gradient inside bloom
    float gradT = clamp(distortedDist / max(radius, 0.001), 0.0, 1.0);
    vec3 bloomColor = mix(cCream, cAmber, smoothstep(0.0, 0.45, gradT));
    bloomColor = mix(bloomColor, cCoral, smoothstep(0.4, 0.8, gradT));
    bloomColor = mix(bloomColor, cEdge, smoothstep(0.75, 1.0, gradT));

    // Horizon line lingering at the bottom
    float horizon = smoothstep(0.008, 0.0, abs(v_uv.y - 0.015)) * smoothstep(0.6, 1.0, u_progress);

    // Final color composition
    vec3 col = mix(cNight, bloomColor, inkMask);
    col += edgeGlow * cEdge * 0.75;
    col += horizon * cCoral * (1.0 + 0.3 * sin(u_time * 2.0));

    // Film grain
    float grain = (hash(gl_FragCoord.xy * 0.6 + vec2(u_time * 0.01)) - 0.5) * 0.025;
    col += grain;

    fragColor = vec4(col, 1.0);
}
`;

export default function HeroReveal({
  eyebrow = "ASK ANYTHING",
  title = "Voice-Enabled RAG",
  subtitle = "Speak the question. Get a grounded, cited answer — end to end, under 200ms.",
  onDawnComplete,
  onStartTalking,
}) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const programRef = useRef(null);
  const animFrameRef = useRef(null);
  const progressRef = useRef(0);
  const startTimeRef = useRef(null);

  const [hasRevealed, setHasRevealed] = useState(false);
  const [animProgress, setAnimProgress] = useState(0);
  const [webGLSupported, setWebGLSupported] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);

  // Initialize WebGL2
  const initWebGL = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return false;

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setReducedMotion(true);
      setAnimProgress(1);
      setHasRevealed(true);
      return false;
    }

    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
    if (!gl) {
      setWebGLSupported(false);
      setAnimProgress(1);
      setHasRevealed(true);
      return false;
    }

    glRef.current = gl;

    // Compile Vertex Shader
    const vShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vShader, VERTEX_SHADER_SRC);
    gl.compileShader(vShader);
    if (!gl.getShaderParameter(vShader, gl.COMPILE_STATUS)) {
      console.error("Vertex Shader Error:", gl.getShaderInfoLog(vShader));
      setWebGLSupported(false);
      return false;
    }

    // Compile Fragment Shader
    const fShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fShader, FRAGMENT_SHADER_SRC);
    gl.compileShader(fShader);
    if (!gl.getShaderParameter(fShader, gl.COMPILE_STATUS)) {
      console.error("Fragment Shader Error:", gl.getShaderInfoLog(fShader));
      setWebGLSupported(false);
      return false;
    }

    // Link Program
    const program = gl.createProgram();
    gl.attachShader(program, vShader);
    gl.attachShader(program, fShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program Link Error:", gl.getProgramInfoLog(program));
      setWebGLSupported(false);
      return false;
    }

    programRef.current = program;

    // Full-screen Quad geometry
    const positions = new Float32Array([
      -1.0, -1.0,
       1.0, -1.0,
      -1.0,  1.0,
      -1.0,  1.0,
       1.0, -1.0,
       1.0,  1.0,
    ]);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    return true;
  }, []);

  // Resize canvas according to container
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const gl = glRef.current;
    if (!canvas || !gl) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.floor(rect.width * dpr);
    const height = Math.floor(rect.height * dpr);

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
  }, []);

  // Run the render animation loop
  const triggerReveal = useCallback(() => {
    if (reducedMotion || !webGLSupported) {
      setAnimProgress(1);
      setHasRevealed(true);
      return;
    }

    cancelAnimationFrame(animFrameRef.current);
    progressRef.current = 0;
    setAnimProgress(0);
    startTimeRef.current = performance.now();
    const DURATION = 2100; // ms

    const render = (now) => {
      const gl = glRef.current;
      const program = programRef.current;
      const canvas = canvasRef.current;

      if (!gl || !program || !canvas) return;

      resizeCanvas();

      const elapsed = now - startTimeRef.current;
      let p = Math.min(elapsed / DURATION, 1.0);

      // Smooth cubic-exponential ease-out
      const easeOut = 1 - Math.pow(1 - p, 3.2);
      progressRef.current = easeOut;
      setAnimProgress(easeOut);

      gl.useProgram(program);

      const uProgress = gl.getUniformLocation(program, "u_progress");
      const uTime = gl.getUniformLocation(program, "u_time");
      const uRes = gl.getUniformLocation(program, "u_resolution");

      gl.uniform1f(uProgress, easeOut);
      gl.uniform1f(uTime, now * 0.001);
      gl.uniform2f(uRes, canvas.width, canvas.height);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      if (p < 1.0) {
        animFrameRef.current = requestAnimationFrame(render);
      } else {
        setHasRevealed(true);
        if (onDawnComplete) onDawnComplete();
      }
    };

    animFrameRef.current = requestAnimationFrame(render);
  }, [reducedMotion, webGLSupported, resizeCanvas, onDawnComplete]);

  // IntersectionObserver trigger at 40% visibility
  useEffect(() => {
    const isSupported = initWebGL();
    if (!isSupported) return;

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !hasRevealed) {
            triggerReveal();
          }
        });
      },
      { threshold: 0.4 }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [initWebGL, resizeCanvas, triggerReveal, hasRevealed]);

  return (
    <section className="hero-reveal-container" ref={containerRef}>
      {/* WebGL Canvas Background */}
      <canvas ref={canvasRef} className="hero-webgl-canvas" />

      {/* Fallback for reduced motion */}
      {(!webGLSupported || reducedMotion) && <div className="hero-fallback-backdrop" />}

      {/* Stitch Reveal Card Panel */}
      <div
        className="hero-reveal-card"
        style={{
          opacity: Math.max(0, (animProgress - 0.3) / 0.7),
          transform: `scale(${0.96 + 0.04 * animProgress}) translateY(${(1 - animProgress) * 14}px)`,
        }}
      >
        <div className="hero-eyebrow-pill">
          <span className="eyebrow-dot" />
          <span className="font-mono">{eyebrow}</span>
        </div>

        <h1 className="hero-headline">{title}</h1>

        <p className="hero-subhead">{subtitle}</p>

        <div className="hero-actions-row">
          <a
            href="#voice-rag-console"
            className="btn-start-talking"
            onClick={onStartTalking}
          >
            <Mic size={18} />
            <span>Start Talking</span>
          </a>

          <button
            className="btn-replay-reveal"
            onClick={triggerReveal}
            title="Replay WebGL dawn reveal bloom"
          >
            <RefreshCw size={15} className="icon-spin-hover" />
            <span>Replay Reveal</span>
          </button>
        </div>

        {/* Feature Micro-Badges */}
        <div className="hero-chips-row">
          <div className="chip-meta">
            <Sparkles size={13} className="text-amber" />
            <span>Pure WebGL2 FBM</span>
          </div>
          <div className="chip-meta">
            <Zap size={13} className="text-coral" />
            <span className="font-mono">P50 &lt; 90ms</span>
          </div>
          <div className="chip-meta">
            <ShieldCheck size={13} className="text-emerald" />
            <span>Grounded Guardrails</span>
          </div>
        </div>
      </div>

      {/* Horizon base accent */}
      <div
        className="hero-horizon-bar"
        style={{
          opacity: Math.min(1, animProgress * 1.3),
          transform: `scaleX(${Math.min(1, animProgress * 1.05)})`,
        }}
      />
    </section>
  );
}
