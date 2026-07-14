import React, { useState, useEffect, useCallback } from 'react'
import { LABEL_COLOR as BASE_COLOR, LABEL_PHRASE as PHRASE } from '../labels.js'

const baseOf = (lbl) => (lbl || '').replace('ml_', '')

export default function DemoView({ onLogin }) {
  const [clips, setClips]           = useState([])
  const [feed, setFeed]             = useState([])
  const [agentLive, setAgentLive]   = useState(false)
  const [showLogin, setShowLogin]   = useState(false)
  const [password, setPassword]     = useState('')
  const [loginError, setLoginError] = useState('')

  // Load curated highlight clips from the public endpoint. The server returns
  // only non-normal labeled clips, so the full archive is never exposed here.
  useEffect(() => {
    fetch('/api/recordings/highlights')
      .then(r => r.json())
      .then(({ recordings }) => setClips((recordings || []).slice(0, 12)))
      .catch(() => {})
  }, [])

  // Poll whether the agent is actively streaming frames
  useEffect(() => {
    const poll = () =>
      fetch('/api/stream/status')
        .then(r => r.json())
        .then(d => setAgentLive(!!d.live))
        .catch(() => setAgentLive(false))
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [])

  // Poll the live prediction text feed
  const pollFeed = useCallback(() => {
    fetch('/api/predictions')
      .then(r => r.json())
      .then(d => setFeed((d.events || []).slice().reverse()))
      .catch(() => {})
  }, [])

  useEffect(() => {
    pollFeed()
    const id = setInterval(pollFeed, 5000)
    return () => clearInterval(id)
  }, [pollFeed])

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoginError('')
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })
      if (res.ok) {
        onLogin()
      } else {
        const d = await res.json().catch(() => ({}))
        setLoginError(d.error || 'Login failed')
      }
    } catch {
      setLoginError('Server unreachable')
    }
  }

  const fmtTime = (iso) =>
    new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const isLive = agentLive

  return (
    <div className="demo">
      {/* Hero */}
      <section className="demo-hero">
        <h1 className="demo-title">🐇 BunnyCam</h1>
        <p className="demo-tagline">
          A machine-learning pet monitor that watches rabbits and recognizes what
          they're doing (zoomies, yawning, grooming, standing) in real time.
        </p>
        <div className="demo-status">
          <span className={`status-dot ${isLive ? 'live' : ''}`} />
          {isLive ? 'System live, detecting right now' : 'Monitor currently offline'}
        </div>
      </section>

      <div className="demo-cols">
        {/* Live prediction feed (text only — no video exposed) */}
        <section className="demo-feed-card">
          <h2 className="demo-section-title">Live Behavior Feed</h2>
          <p className="demo-note">
            Real-time output from the classifier. The live video stays private —
            this feed proves the model is running without opening a window into anyone's home.
          </p>
          <div className="demo-feed">
            {feed.length === 0 && (
              <div className="demo-feed-empty">No predictions yet. The camera is offline.</div>
            )}
            {feed.map((ev, i) => {
              const base = baseOf(ev.label)
              return (
                <div key={i} className="feed-row">
                  <span className="feed-time">{fmtTime(ev.time)}</span>
                  <span className="feed-dot" style={{ background: BASE_COLOR[base] }} />
                  <span className="feed-msg" style={{ color: BASE_COLOR[base] }}>
                    {PHRASE[base] ?? base}
                  </span>
                  {ev.confidence != null && <span className="feed-conf">{ev.confidence}%</span>}
                </div>
              )
            })}
          </div>
        </section>

        {/* How it works */}
        <section className="demo-about-card">
          <h2 className="demo-section-title">How It Works</h2>
          <ol className="demo-steps">
            <li><b>Capture:</b> a camera watches the bunny pen; motion detection triggers recording.</li>
            <li><b>Embed:</b> frames run through MobileNetV2 (transfer learning) producing 1280-dim features.</li>
            <li><b>Classify:</b> a custom dense network (trained on 150+ hand-labeled clips) maps features to one of five behaviors.</li>
            <li><b>Alert:</b> non-normal behaviors trigger an email with the clip attached, with smart cooldowns.</li>
          </ol>
          <div className="demo-chips">
            {Object.entries(BASE_COLOR).map(([l, c]) => (
              <span key={l} className="demo-chip" style={{ color: c, borderColor: c }}>{l}</span>
            ))}
          </div>
        </section>
      </div>

      {/* Highlight clips */}
      <section className="demo-clips-section">
        <h2 className="demo-section-title">Detection Highlights</h2>
        <div className="demo-clips-grid">
          {clips.length === 0 && <div className="demo-feed-empty">No highlight clips published yet.</div>}
          {clips.map(clip => {
            const base = baseOf(clip.label)
            return (
              <div key={clip.filename} className="demo-clip">
                <video controls muted preload="metadata" src={`/recordings/${clip.filename}`} />
                <span
                  className="demo-clip-label"
                  style={{ color: BASE_COLOR[base], borderColor: BASE_COLOR[base] }}
                >
                  {base}
                </span>
              </div>
            )
          })}
        </div>
      </section>

      {/* Admin login */}
      <footer className="demo-footer">
        {!showLogin ? (
          <button className="demo-login-link" onClick={() => setShowLogin(true)}>
            Owner login
          </button>
        ) : (
          <form className="demo-login-form" onSubmit={handleLogin}>
            <input
              type="password"
              placeholder="Admin password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              autoFocus
            />
            <button type="submit">Log in</button>
            {loginError && <span className="demo-login-error">{loginError}</span>}
          </form>
        )}
      </footer>

      <style>{`
        .demo {
          max-width: 980px;
          margin: 0 auto;
          padding: 40px 24px 60px;
          display: flex;
          flex-direction: column;
          gap: 28px;
        }

        .demo-hero { text-align: center; }
        .demo-title {
          font-family: var(--font-display);
          font-size: 40px;
          color: var(--text-primary);
          margin-bottom: 8px;
        }
        .demo-tagline {
          color: var(--text-secondary);
          font-size: 14px;
          max-width: 560px;
          margin: 0 auto 14px;
          line-height: 1.6;
        }
        .demo-status {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-size: 12px;
          color: var(--text-muted);
          border: 1px solid var(--border);
          border-radius: 99px;
          padding: 5px 14px;
          background: var(--bg-card);
        }
        .status-dot {
          width: 8px; height: 8px; border-radius: 50%;
          background: var(--text-muted);
        }
        .status-dot.live {
          background: #7dff7d;
          animation: pulse-dot 1.4s ease-in-out infinite;
        }
        @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:.35} }

        .demo-cols {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
        }
        @media (max-width: 720px) { .demo-cols { grid-template-columns: 1fr; } }

        .demo-feed-card, .demo-about-card {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 18px;
        }

        .demo-section-title {
          font-family: var(--font-display);
          font-size: 15px;
          color: var(--text-primary);
          margin-bottom: 8px;
        }
        .demo-note {
          font-size: 11px;
          color: var(--text-muted);
          line-height: 1.5;
          margin-bottom: 12px;
        }

        .demo-feed {
          max-height: 240px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 2px;
          font-family: var(--font-mono);
        }
        .demo-feed-empty { color: var(--text-muted); font-size: 12px; padding: 16px 0; }
        .feed-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 11px; }
        .feed-time { color: var(--text-muted); font-size: 10px; min-width: 64px; }
        .feed-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
        .feed-msg { flex: 1; }
        .feed-conf { color: var(--text-muted); font-size: 10px; }

        .demo-steps {
          padding-left: 18px;
          display: flex; flex-direction: column; gap: 8px;
          color: var(--text-secondary);
          font-size: 12px; line-height: 1.5;
        }
        .demo-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 14px; }
        .demo-chip {
          font-size: 10px; padding: 2px 9px;
          border: 1px solid; border-radius: 99px;
          font-family: var(--font-display);
        }

        .demo-clips-section {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 18px;
        }
        .demo-clips-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
          gap: 12px;
          margin-top: 10px;
        }
        .demo-clip { position: relative; }
        .demo-clip video {
          width: 100%;
          aspect-ratio: 16/9;
          object-fit: cover;
          border-radius: var(--radius);
          border: 1px solid var(--border);
          display: block;
        }
        .demo-clip-label {
          position: absolute;
          top: 6px; left: 6px;
          font-size: 9px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-family: var(--font-display);
          background: rgba(10,10,10,0.78);
          border: 1px solid;
          border-radius: 3px;
          padding: 2px 7px;
          pointer-events: none;
        }

        .demo-footer { text-align: center; padding-top: 8px; }
        .demo-login-link {
          background: none;
          color: var(--text-muted);
          font-size: 11px;
          text-decoration: underline;
          cursor: pointer;
        }
        .demo-login-link:hover { color: var(--text-secondary); }

        .demo-login-form {
          display: inline-flex;
          gap: 8px;
          align-items: center;
        }
        .demo-login-form input {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          color: var(--text-primary);
          padding: 6px 10px;
          font-size: 12px;
          outline: none;
        }
        .demo-login-form input:focus { border-color: var(--accent); }
        .demo-login-form button {
          background: var(--accent);
          color: #000;
          border-radius: var(--radius);
          padding: 6px 14px;
          font-size: 12px;
          font-family: var(--font-display);
          cursor: pointer;
        }
        .demo-login-error { color: #ff6b6b; font-size: 11px; }
      `}</style>
    </div>
  )
}
