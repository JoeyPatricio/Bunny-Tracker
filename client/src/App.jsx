import React, { useState, useEffect, useCallback, useRef } from 'react'
import VideoFeed from './components/VideoFeed.jsx'
import Controls from './components/Controls.jsx'
import ActivityLog from './components/ActivityLog.jsx'
import RecordingGallery from './components/RecordingGallery.jsx'
import LabelingStudio from './components/LabelingStudio.jsx'
import DemoView from './components/DemoView.jsx'
import AgentFeed from './components/AgentFeed.jsx'
import HighlightsManager from './components/HighlightsManager.jsx'
import { lazy, Suspense } from 'react'
const TrainingStudio = lazy(() => import('./components/TrainingStudio.jsx'))
import { useWebcam } from './hooks/useWebcam.js'
import { useMotion } from './hooks/useMotion.js'
import { useInference } from './hooks/useInference.js'
import { LABEL_COLOR, LABEL_PHRASE } from './labels.js'

export default function App() {
  const [authed, setAuthed]               = useState(null) // null = checking | false | true
  const [agentLive, setAgentLive]         = useState(false)
  const [monitorOn, setMonitorOn]         = useState(null) // null until fetched
  const [emailOn, setEmailOn]             = useState(null)
  const [tab, setTab]                     = useState('camera') // 'camera' | 'label' | 'train'
  const [inferenceEnabled, setInferenceEnabled] = useState(
    () => localStorage.getItem('bunnycam.aiEnabled') === 'true'
  )
  const [events, setEvents]               = useState([])
  const [sensitivity, setSensitivity]     = useState(30)
  const [recordingTick, setRecordingTick] = useState(0)
  const [autoRecordEnabled, setAutoRecordEnabled] = useState(
    () => localStorage.getItem('bunnycam.autoRecord') === 'true'
  )
  const [isRecording, setIsRecording]     = useState(false)
  const eventIdRef = useRef(0)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const recordTimeoutRef = useRef(null)
  const lastPredictionLabelRef = useRef(null)
  const currentPredictionRef = useRef(null)

  // ── Webcam ──────────────────────────────────────────────
  const {
    videoRef, isActive, error,
    devices, selectedDevice,
    startCamera, stopCamera, switchCamera,
  } = useWebcam()

  const getSupportedMimeType = useCallback(() => {
    if (typeof window === 'undefined' || !window.MediaRecorder) return ''
    const candidateTypes = [
      'video/webm; codecs=vp9',
      'video/webm; codecs=vp8',
      'video/webm',
    ]
    return candidateTypes.find(type => MediaRecorder.isTypeSupported(type)) || ''
  }, [])

  const uploadRecording = useCallback(async () => {
    if (chunksRef.current.length === 0) return

    const mimeType = getSupportedMimeType() || 'video/webm'
    const blob = new Blob(chunksRef.current, { type: mimeType })
    const formData = new FormData()
    formData.append('video', blob, `recording-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`)

    try {
      const res  = await fetch('/api/recordings', { method: 'POST', body: formData })
      const data = await res.json()

      // Notify if AI predicts a non-normal behavior. Clips are left UNLABELED
      // so they can be reviewed and labeled by hand (e.g. as `absent`).
      const pred = currentPredictionRef.current
      if (pred && pred.warm && data.filename && pred.label !== 'normal') {
        fetch('/api/notify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            label:      pred.label,
            confidence: pred.confidence,
            filename:   data.filename,
          }),
        }).catch(() => {})
      }

      setRecordingTick(t => t + 1)
    } catch (err) {
      console.error('Recording upload failed:', err)
    } finally {
      chunksRef.current = []
    }
  }, [getSupportedMimeType])

  const stopRecording = useCallback(() => {
    if (recordTimeoutRef.current) {
      window.clearTimeout(recordTimeoutRef.current)
      recordTimeoutRef.current = null
    }

    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }
  }, [])

  const scheduleRecordingStop = useCallback(() => {
    if (recordTimeoutRef.current) {
      window.clearTimeout(recordTimeoutRef.current)
    }

    recordTimeoutRef.current = window.setTimeout(() => {
      stopRecording()
    }, 10000)
  }, [stopRecording])

  const startRecording = useCallback(() => {
    const video = videoRef.current
    if (!video || !video.srcObject || !window.MediaRecorder) return

    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      scheduleRecordingStop()
      return
    }

    const mimeType = getSupportedMimeType()
    const options = mimeType ? { mimeType } : undefined
    const recorder = new MediaRecorder(video.srcObject, options)

    chunksRef.current = []
    recorderRef.current = recorder

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data)
      }
    }

    recorder.onstop = () => {
      setIsRecording(false)
      recorderRef.current = null
      uploadRecording()
    }

    recorder.start()
    setIsRecording(true)
    scheduleRecordingStop()
  }, [getSupportedMimeType, scheduleRecordingStop, uploadRecording, videoRef])

  const handleManualRecord = useCallback(() => {
    if (!isRecording) {
      startRecording()
    }
  }, [isRecording, startRecording])

  const handleToggleAutoRecord = useCallback(() => {
    setAutoRecordEnabled(prev => !prev)
  }, [])

  const handleMotion = useCallback(({ level, timestamp }) => {
    const id = ++eventIdRef.current
    setEvents(prev => [
      ...prev.slice(-199), // keep last 200 events
      {
        id,
        timestamp,
        message: 'Bunnies moving!',
        level,
      }
    ])

    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('BunnyCam 🐇', { body: 'Motion detected!', silent: true })
    }

    if (autoRecordEnabled) {
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        scheduleRecordingStop()
      } else {
        startRecording()
      }
    }
  }, [autoRecordEnabled, scheduleRecordingStop, startRecording])

  const {
    motionLevel, isMotion,
    motionEnabled, toggleMotion,
    overlayCanvasRef,
  } = useMotion({ videoRef, isActive, sensitivity, onMotion: handleMotion })

  const { status: inferenceStatus, prediction, modelExists } = useInference({
    videoRef,
    isActive,
    enabled: inferenceEnabled,
  })

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
    stopCamera()
    setAuthed(false)
  }, [stopCamera])

  // Keep current prediction accessible inside uploadRecording callback
  useEffect(() => { currentPredictionRef.current = prediction }, [prediction])

  // Persist toggles so a page reload doesn't disarm overnight monitoring
  useEffect(() => {
    localStorage.setItem('bunnycam.aiEnabled', String(inferenceEnabled))
  }, [inferenceEnabled])
  useEffect(() => {
    localStorage.setItem('bunnycam.autoRecord', String(autoRecordEnabled))
  }, [autoRecordEnabled])

  // Log to activity log + trigger SMS when the predicted label changes
  useEffect(() => {
    if (!prediction) { lastPredictionLabelRef.current = null; return }
    if (prediction.label === lastPredictionLabelRef.current) return
    lastPredictionLabelRef.current = prediction.label

    const id = ++eventIdRef.current
    setEvents(prev => [
      ...prev.slice(-199),
      {
        id,
        timestamp: new Date(),
        type: 'prediction',
        message: LABEL_PHRASE[prediction.label] ?? prediction.label,
        confidence: prediction.confidence,
        color: LABEL_COLOR[prediction.label],
      },
    ])

    // Publish to the server's public text-only demo feed, but not warmup labels
    // (sub-window std features are unreliable until the embedding window fills)
    if (prediction.warm) {
      fetch('/api/predictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: prediction.label, confidence: prediction.confidence }),
      }).catch(() => {})
    }

    // SMS/email is fired from uploadRecording once a clip is saved (with attachment)
  }, [prediction])

  // ── Notification permission ───────────────────────────────
  const requestNotifications = useCallback(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [])

  const handleStart = useCallback(() => {
    requestNotifications()
    startCamera()
  }, [startCamera, requestNotifications])

  useEffect(() => {
    return () => {
      if (recordTimeoutRef.current) {
        window.clearTimeout(recordTimeoutRef.current)
      }

      const recorder = recorderRef.current
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop()
      }
    }
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
          ) : isActive ? (
            <span className="status-active">● watching</span>
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
      {tab === 'train' && (
        <Suspense fallback={<div style={{ padding: 32, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>Loading TensorFlow.js…</div>}>
          <TrainingStudio />
        </Suspense>
      )}

      {/* Main layout — only mounted when on camera tab */}
      <main className="app-main" style={{ display: tab === 'camera' ? 'grid' : 'none' }}>
        {/* Left col: video + controls */}
        <section className="col-main">
          {agentLive ? (
            <AgentFeed />
          ) : (
            <VideoFeed
              videoRef={videoRef}
              overlayCanvasRef={overlayCanvasRef}
              isActive={isActive}
              isMotion={isMotion}
              error={error}
              prediction={prediction}
              inferenceStatus={inferenceStatus}
            />
          )}

          <div className="controls-card" style={{ display: agentLive ? 'none' : undefined }}>
            <Controls
              isActive={isActive}
              motionEnabled={motionEnabled}
              isRecording={isRecording}
              autoRecordEnabled={autoRecordEnabled}
              motionLevel={motionLevel}
              sensitivity={sensitivity}
              devices={devices}
              selectedDevice={selectedDevice}
              onStart={handleStart}
              onStop={stopCamera}
              onRecord={handleManualRecord}
              onToggleAutoRecord={handleToggleAutoRecord}
              onToggleMotion={toggleMotion}
              onSensitivityChange={setSensitivity}
              onSwitchCamera={switchCamera}
              inferenceEnabled={inferenceEnabled}
              inferenceStatus={inferenceStatus}
              modelExists={modelExists}
              onToggleInference={() => setInferenceEnabled(e => !e)}
            />
          </div>
        </section>

        {/* Right col: log + galleries */}
        <aside className="col-side">
          <ActivityLog
            events={events}
            onClear={() => setEvents([])}
          />
          <RecordingGallery refreshTrigger={recordingTick} />
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

        .controls-card {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 16px;
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
