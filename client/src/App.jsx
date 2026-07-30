import React, { useState, useEffect, useCallback } from 'react'
import ActivityLog from './components/ActivityLog.jsx'
import RecordingGallery from './components/RecordingGallery.jsx'
import LabelingStudio from './components/LabelingStudio.jsx'
import DemoView from './components/DemoView.jsx'
import AgentFeed from './components/AgentFeed.jsx'
import HighlightsManager from './components/HighlightsManager.jsx'
import TrainingStudio from './components/TrainingStudio.jsx'

export default function App() {
  const [authed, setAuthed]       = useState(null) // null = checking | false | true
  const [agentLive, setAgentLive] = useState(false)
  const [monitorOn, setMonitorOn] = useState(null) // null until fetched
  const [emailOn, setEmailOn]     = useState(null)
  const [tab, setTab]             = useState('camera') // 'camera' | 'label' | 'highlights' | 'train'
  // Bumped whenever something saves a clip, so RecordingGallery refetches.
  // Its effect only reruns on mount otherwise, which left saved clips invisible
  // until a full page reload.
  const [galleryTick, setGalleryTick] = useState(0)

  // Poll whether the headless camera agent is streaming
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

  // Check admin session on load
  useEffect(() => {
    fetch('/api/auth/check')
      .then(r => r.json())
      .then(d => setAuthed(!!d.authed))
      .catch(() => setAuthed(false))
  }, [])

  // Monitoring + email-alert state
  useEffect(() => {
    fetch('/api/monitor')
      .then(r => r.json())
      .then(d => { setMonitorOn(!!d.enabled); setEmailOn(!!d.emailAlerts) })
      .catch(() => {})
  }, [])

  const toggleMonitor = useCallback(async () => {
    const next = !monitorOn
    setMonitorOn(next) // optimistic
    try {
      const res = await fetch('/api/monitor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: next }),
      })
      if (!res.ok) setMonitorOn(!next)
    } catch {
      setMonitorOn(!next)
    }
  }, [monitorOn])

  const toggleEmail = useCallback(async () => {
    const next = !emailOn
    setEmailOn(next) // optimistic
    try {
      const res = await fetch('/api/monitor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emailAlerts: next }),
      })
      if (!res.ok) setEmailOn(!next)
    } catch {
      setEmailOn(!next)
    }
  }, [emailOn])

  const handleLogout = useCallback(async () => {
    await fetch('/api/auth/logout', { method: 'POST' }).catch(() => {})
    setAuthed(false)
  }, [])

  // ── Auth gate ───────────────────────────────────────────
  if (authed === null) {
    return <div style={{ padding: 48, textAlign: 'center', color: 'var(--text-muted)' }}>…</div>
  }
  if (!authed) {
    return <DemoView onLogin={() => setAuthed(true)} />
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-logo">
          <span className="logo-emoji">🐇</span>
          <span className="logo-text">BunnyCam</span>
        </div>

        <nav className="header-tabs">
          <button
            className={`header-tab ${tab === 'camera' ? 'tab-active' : ''}`}
            onClick={() => setTab('camera')}
          >
            Camera
          </button>
          <button
            className={`header-tab ${tab === 'label' ? 'tab-active' : ''}`}
            onClick={() => setTab('label')}
          >
            Label Studio
          </button>
          <button
            className={`header-tab ${tab === 'highlights' ? 'tab-active' : ''}`}
            onClick={() => setTab('highlights')}
          >
            Highlights
          </button>
          <button
            className={`header-tab ${tab === 'train' ? 'tab-active' : ''}`}
            onClick={() => setTab('train')}
          >
            Training
          </button>
        </nav>

        <div className="header-status">
          {monitorOn !== null && (
            <button
              className={`monitor-toggle ${monitorOn ? 'monitor-on' : 'monitor-off'}`}
              onClick={toggleMonitor}
              title={monitorOn ? 'Turn monitoring off (releases camera)' : 'Turn monitoring on'}
            >
              {monitorOn ? '● Monitoring ON' : '○ Monitoring OFF'}
            </button>
          )}
          {emailOn !== null && (
            <button
              className={`monitor-toggle ${emailOn ? 'monitor-on' : 'monitor-off'}`}
              onClick={toggleEmail}
              title={emailOn ? 'Turn email alerts off' : 'Turn email alerts on'}
            >
              {emailOn ? '✉ Email ON' : '✉ Email OFF'}
            </button>
          )}
          {agentLive ? (
            <span className="status-active">● agent live</span>
          ) : (
            <span className="status-idle">○ idle</span>
          )}
          <button className="logout-btn" onClick={handleLogout} title="Log out to demo view">
            Log out
          </button>
        </div>
      </header>

      {tab === 'label' && <LabelingStudio />}
      {tab === 'highlights' && <HighlightsManager />}
      {tab === 'train' && <TrainingStudio />}

      {/* Main layout — only mounted when on camera tab */}
      <main className="app-main" style={{ display: tab === 'camera' ? 'grid' : 'none' }}>
        {/* Left col: agent video feed */}
        <section className="col-main">
          <AgentFeed onClipSaved={() => setGalleryTick(n => n + 1)} />
        </section>

        {/* Right col: log + galleries */}
        <aside className="col-side">
          <ActivityLog />
          <RecordingGallery refreshTrigger={galleryTick} />
        </aside>
      </main>

      <style>{`
        .app {
          display: flex;
          flex-direction: column;
          min-height: 100vh;
          padding: 0 0 32px;
        }

        /* Header */
        .app-header {
          display: flex;
          align-items: center;
          gap: 20px;
          padding: 0 28px;
          border-bottom: 1px solid var(--border);
          background: var(--bg-surface);
          height: 52px;
        }

        .header-logo {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-shrink: 0;
        }

        .header-tabs {
          display: flex;
          align-items: stretch;
          gap: 0;
          height: 100%;
          margin-left: 8px;
        }

        .header-tab {
          height: 100%;
          padding: 0 18px;
          font-size: 12px;
          font-family: var(--font-mono);
          letter-spacing: 0.06em;
          color: var(--text-muted);
          background: transparent;
          border: none;
          border-bottom: 2px solid transparent;
          cursor: pointer;
          transition: all 0.15s;
        }

        .header-tab:hover { color: var(--text-secondary); }

        .tab-active {
          color: var(--text-primary) !important;
          border-bottom-color: var(--accent) !important;
        }

        .header-status {
          margin-left: auto;
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .logout-btn {
          background: none;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          color: var(--text-muted);
          font-size: 10px;
          padding: 3px 9px;
          cursor: pointer;
          transition: all 0.15s;
        }
        .logout-btn:hover { color: var(--text-secondary); border-color: var(--border-light); }

        .monitor-toggle {
          background: none;
          border: 1px solid;
          border-radius: 99px;
          font-size: 10px;
          letter-spacing: 0.06em;
          padding: 3px 11px;
          cursor: pointer;
          font-family: var(--font-display);
          transition: all 0.15s;
        }
        .monitor-on  { color: #7dff7d; border-color: #7dff7d66; }
        .monitor-on:hover  { background: #7dff7d14; }
        .monitor-off { color: var(--text-muted); border-color: var(--border); }
        .monitor-off:hover { color: var(--text-secondary); }

        .logo-emoji { font-size: 22px; }

        .logo-text {
          font-family: var(--font-display);
          font-size: 22px;
          font-weight: 600;
          color: var(--text-primary);
          letter-spacing: 0.02em;
        }

        .status-active {
          color: var(--green);
          font-size: 11px;
          letter-spacing: 0.08em;
        }

        .status-idle {
          color: var(--text-muted);
          font-size: 11px;
          letter-spacing: 0.08em;
        }

        /* Layout */
        .app-main {
          display: grid;
          grid-template-columns: 1fr 320px;
          gap: 20px;
          padding: 20px 28px 0;
          flex: 1;
          align-items: start;
        }

        .col-main {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .col-side {
          display: flex;
          flex-direction: column;
          gap: 16px;
          position: sticky;
          top: 20px;
        }

        /* Responsive — tablet and below: single column */
        @media (max-width: 900px) {
          .app-main {
            grid-template-columns: 1fr;
            padding: 16px 16px 0;
          }
          .col-side {
            position: static;
          }
        }

        /* Mobile — stack the header into rows so nothing overflows */
        @media (max-width: 640px) {
          .app-header {
            height: auto;
            flex-direction: column;
            align-items: stretch;
            gap: 10px;
            padding: 12px 14px;
          }

          .header-logo { justify-content: center; }
          .logo-emoji  { font-size: 20px; }
          .logo-text   { font-size: 20px; }

          .header-tabs {
            margin-left: 0;
            width: 100%;
            height: 42px;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
          }
          .header-tab {
            flex: 1 0 auto;
            text-align: center;
            padding: 0 14px;
            font-size: 11px;
            white-space: nowrap;
            border-bottom: none;
            border-right: 1px solid var(--border);
          }
          .header-tab:last-child { border-right: none; }
          .tab-active {
            background: var(--bg-card);
            border-bottom-color: transparent !important;
          }

          .header-status {
            margin-left: 0;
            flex-wrap: wrap;
            justify-content: center;
            row-gap: 8px;
          }

          .app-main { padding: 14px 12px 0; gap: 14px; }
        }
      `}</style>
    </div>
  )
}
