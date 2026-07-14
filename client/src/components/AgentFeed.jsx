import React, { useState, useEffect, useRef } from 'react'
import { LABEL_COLOR as BASE_COLOR, LABEL_PHRASE as PHRASE } from '../labels.js'

/**
 * AgentFeed
 * Live MJPEG view from the headless camera agent, with the agent's latest
 * prediction overlaid. Replaces the browser-webcam view when the agent runs.
 */
export default function AgentFeed() {
  const [latest, setLatest]     = useState(null) // { label, confidence, time }
  const [recording, setRecording] = useState(false)
  const [flash, setFlash]       = useState('')
  // Two fullscreen modes: native (desktop, Android) and a CSS overlay fallback
  // for iOS Safari, which only allows native fullscreen on <video> elements —
  // our MJPEG feed is an <img>, so el.requestFullscreen is unavailable there.
  const [nativeFull, setNativeFull] = useState(false)
  const [cssFull, setCssFull]       = useState(false)
  const isFull = nativeFull || cssFull
  const containerRef = useRef(null)

  const toggleFullscreen = () => {
    const el = containerRef.current
    if (!el) return
    if (nativeFull) { document.exitFullscreen?.(); return }
    if (cssFull)    { setCssFull(false); return }
    if (el.requestFullscreen && document.fullscreenEnabled) {
      el.requestFullscreen().catch(() => setCssFull(true)) // fall back if it rejects
    } else {
      setCssFull(true) // iOS / unsupported — use the CSS overlay
    }
  }

  // Keep native state in sync (e.g. exiting via Esc or the system gesture)
  useEffect(() => {
    const onChange = () => setNativeFull(document.fullscreenElement === containerRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  // Esc exits the CSS-overlay mode (native fullscreen handles its own Esc)
  useEffect(() => {
    if (!cssFull) return
    const onKey = (e) => { if (e.key === 'Escape') setCssFull(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cssFull])

  const recordNow = async () => {
    setRecording(true)
    try {
      const res = await fetch('/api/recordings/grab', { method: 'POST' })
      const d   = await res.json()
      setFlash(res.ok ? '✓ Saved clip' : `⚠ ${d.error || 'Failed'}`)
    } catch {
      setFlash('⚠ Failed')
    } finally {
      setRecording(false)
      setTimeout(() => setFlash(''), 2500)
    }
  }

  // Poll the prediction feed for the newest entry
  useEffect(() => {
    const poll = () =>
      fetch('/api/predictions')
        .then(r => r.json())
        .then(d => {
          const ev = (d.events || []).at(-1)
          if (ev) setLatest(ev)
        })
        .catch(() => {})
    poll()
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [])

  const base = latest ? latest.label.replace('ml_', '') : null

  return (
    <div className={`agent-feed${cssFull ? ' css-full' : ''}`} ref={containerRef}>
      <img className="agent-stream" src="/api/stream/live" alt="Live bunny cam (agent)" />

      <div className="agent-badge">
        <span className="agent-dot" />
        AGENT LIVE
      </div>

      <button className="agent-record-btn" onClick={recordNow} disabled={recording} title="Save the last ~12s as a clip">
        <span className="agent-record-dot" />
        {recording ? 'Saving…' : 'Record'}
      </button>

      {flash && <div className="agent-flash">{flash}</div>}

      {latest && (
        <div className="agent-pred" style={{ borderColor: BASE_COLOR[base] }}>
          <span className="agent-pred-label" style={{ color: BASE_COLOR[base] }}>
            {PHRASE[base] ?? base}
          </span>
          <span className="agent-pred-conf">{latest.confidence}%</span>
        </div>
      )}

      <button
        className="agent-fs-btn"
        onClick={toggleFullscreen}
        title={isFull ? 'Exit fullscreen' : 'View fullscreen'}
        aria-label={isFull ? 'Exit fullscreen' : 'View fullscreen'}
      >
        {isFull ? (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 3v6H3M21 9h-6V3M15 21v-6h6M3 15h6v6"/></svg>
        ) : (
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/></svg>
        )}
        <span className="agent-fs-text">{isFull ? 'Exit' : 'Fullscreen'}</span>
      </button>

      <style>{`
        .agent-feed {
          position: relative;
          width: 100%;
          aspect-ratio: 16 / 9;
          background: var(--bg-deep);
          border-radius: var(--radius-lg);
          overflow: hidden;
          border: 1px solid var(--border);
        }

        .agent-stream {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
        }

        .agent-badge {
          position: absolute;
          top: 12px;
          left: 12px;
          background: rgba(15, 13, 10, 0.75);
          backdrop-filter: blur(6px);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 3px 8px;
          font-size: 10px;
          letter-spacing: 0.15em;
          color: var(--text-secondary);
          display: flex;
          align-items: center;
          gap: 5px;
        }

        .agent-dot {
          width: 6px; height: 6px; border-radius: 50%;
          background: #7dff7d;
          animation: agent-blink 1.4s ease-in-out infinite;
        }
        @keyframes agent-blink { 0%,100%{opacity:1} 50%{opacity:.3} }

        .agent-record-btn {
          position: absolute;
          top: 12px;
          right: 12px;
          display: flex;
          align-items: center;
          gap: 6px;
          background: rgba(15, 13, 10, 0.78);
          backdrop-filter: blur(6px);
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 5px 12px;
          font-size: 11px;
          color: var(--text-secondary);
          cursor: pointer;
          font-family: var(--font-display);
          letter-spacing: 0.04em;
          transition: all 0.15s;
        }
        .agent-record-btn:hover:not(:disabled) { border-color: var(--red); color: #fff; }
        .agent-record-btn:disabled { opacity: 0.6; cursor: default; }
        .agent-record-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--red, #ff4444); }

        .agent-flash {
          position: absolute;
          top: 46px;
          right: 12px;
          background: rgba(15, 13, 10, 0.85);
          border: 1px solid var(--border);
          border-radius: 5px;
          padding: 4px 10px;
          font-size: 11px;
          color: var(--text-secondary);
        }

        .agent-pred {
          position: absolute;
          bottom: 12px;
          left: 12px;
          background: rgba(15, 13, 10, 0.82);
          backdrop-filter: blur(8px);
          border: 1px solid;
          border-radius: 6px;
          padding: 6px 12px;
          display: flex;
          align-items: baseline;
          gap: 8px;
        }
        .agent-pred-label {
          font-family: var(--font-display);
          font-size: 15px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }
        .agent-pred-conf {
          font-size: 11px;
          color: var(--text-muted);
          font-family: var(--font-mono);
        }

        .agent-fs-btn {
          position: absolute;
          bottom: 12px;
          right: 12px;
          display: flex;
          align-items: center;
          gap: 6px;
          background: rgba(15, 13, 10, 0.78);
          backdrop-filter: blur(6px);
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 5px 12px;
          font-size: 11px;
          color: var(--text-secondary);
          cursor: pointer;
          font-family: var(--font-display);
          letter-spacing: 0.04em;
          transition: all 0.15s;
        }
        .agent-fs-btn:hover { border-color: var(--accent); color: #fff; }
        .agent-fs-btn svg { display: block; }

        /* Fullscreen: fill the screen and show the whole frame (no crop) */
        .agent-feed:fullscreen {
          width: 100vw;
          height: 100vh;
          aspect-ratio: auto;
          border-radius: 0;
          border: none;
          background: #000;
        }
        .agent-feed:fullscreen .agent-stream { object-fit: contain; }

        /* CSS-overlay fullscreen fallback (iOS Safari). Uses dvh so it accounts
           for the mobile browser's dynamic toolbar. */
        .agent-feed.css-full {
          position: fixed;
          inset: 0;
          width: 100vw;
          height: 100vh;
          height: 100dvh;
          aspect-ratio: auto;
          z-index: 9999;
          border-radius: 0;
          border: none;
          background: #000;
        }
        .agent-feed.css-full .agent-stream { object-fit: contain; }

        /* On phones the labels would crowd the frame — show icons only */
        @media (max-width: 540px) {
          .agent-fs-text { display: none; }
          .agent-fs-btn,
          .agent-record-btn { padding: 7px 10px; }
        }
      `}</style>
    </div>
  )
}
