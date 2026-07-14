import React from 'react'
import { LABEL_PHRASE } from '../labels.js'

/**
 * VideoFeed
 * Renders the live webcam video with the motion detection overlay canvas on top.
 */
export default function VideoFeed({ videoRef, overlayCanvasRef, isActive, isMotion, error, prediction, inferenceStatus }) {
  return (
    <div className="video-feed-wrapper">
      {isMotion && <div className="motion-ring" aria-hidden="true" />}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`video-element ${isActive ? 'active' : ''}`}
        aria-label="Live bunny cam feed"
      />

      <canvas
        ref={overlayCanvasRef}
        className="motion-overlay"
        aria-hidden="true"
      />

      {!isActive && !error && (
        <div className="feed-placeholder">
          <span className="placeholder-emoji">🐇</span>
          <p className="placeholder-text">Camera is off</p>
          <p className="placeholder-sub">Press Start to begin watching</p>
        </div>
      )}

      {error && (
        <div className="feed-placeholder feed-error">
          <span className="placeholder-emoji">⚠️</span>
          <p className="placeholder-text">Camera error</p>
          <p className="placeholder-sub">{error}</p>
        </div>
      )}

      {isActive && (
        <div className="live-badge" aria-label="Live">
          <span className="live-dot" />
          LIVE
        </div>
      )}

      {inferenceStatus === 'loading' && (
        <div className="inference-loading">loading model…</div>
      )}

      {prediction && (
        <div className="prediction-badge" style={{ borderColor: prediction.color }}>
          <span className="pred-label" style={{ color: prediction.color }}>
            {LABEL_PHRASE[prediction.label] ?? prediction.label}
          </span>
          <span className="pred-conf">{prediction.confidence}%</span>
        </div>
      )}

      <style>{`
        .video-feed-wrapper {
          position: relative;
          width: 100%;
          aspect-ratio: 16 / 9;
          background: var(--bg-deep);
          border-radius: var(--radius-lg);
          overflow: hidden;
          border: 1px solid var(--border);
        }

        .motion-ring {
          position: absolute;
          inset: 0;
          border-radius: var(--radius-lg);
          box-shadow: inset 0 0 0 3px var(--accent);
          animation: ring-pulse 1s ease-in-out infinite;
          pointer-events: none;
          z-index: 10;
        }

        @keyframes ring-pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.4; }
        }

        .video-element {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
          opacity: 0;
          transition: opacity 0.4s ease;
        }

        .video-element.active { opacity: 1; }

        .motion-overlay {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
          z-index: 5;
          mix-blend-mode: screen;
        }

        .feed-placeholder {
          position: absolute;
          inset: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 8px;
          background: var(--bg-deep);
        }

        .feed-error { background: color-mix(in srgb, var(--red-dim) 20%, var(--bg-deep)); }

        .placeholder-emoji { font-size: 48px; }
        .placeholder-text  { color: var(--text-secondary); font-family: var(--font-display); font-size: 18px; }
        .placeholder-sub   { color: var(--text-muted); font-size: 12px; }

        .live-badge {
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
          z-index: 8;
        }

        .live-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--red);
          animation: blink 1.4s ease-in-out infinite;
        }

        @keyframes blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }

        .inference-loading {
          position: absolute;
          bottom: 12px;
          left: 12px;
          background: rgba(15, 13, 10, 0.75);
          backdrop-filter: blur(6px);
          border: 1px solid var(--border);
          border-radius: 4px;
          padding: 4px 10px;
          font-size: 10px;
          color: var(--text-muted);
          letter-spacing: 0.08em;
          z-index: 8;
          animation: pulse-opacity 1.4s ease-in-out infinite;
        }

        @keyframes pulse-opacity {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.4; }
        }

        .prediction-badge {
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
          z-index: 8;
        }

        .pred-label {
          font-family: var(--font-display);
          font-size: 15px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .pred-conf {
          font-size: 11px;
          color: var(--text-muted);
          font-family: var(--font-mono);
        }
      `}</style>
    </div>
  )
}
