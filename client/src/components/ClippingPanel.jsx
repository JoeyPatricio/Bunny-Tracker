import React, { useCallback, useEffect, useRef, useState } from 'react'

/**
 * ClippingPanel
 * Tuning surface for Clipping Mode — the bulk unlabeled-clip harvester.
 *
 * The whole reason this panel exists is that a motion threshold is impossible
 * to pick blind. The agent reports every frame's motion level to /api/motion,
 * so the sparkline below shows the real scene with the current threshold drawn
 * across it: watch the rabbits move, see where the spikes land, put the line
 * under them. Dry run evaluates the threshold and reports what it WOULD have
 * captured without writing anything, which is the cheap way to check a setting
 * before leaving it running overnight.
 *
 * Lives next to the live feed on purpose — tuning means watching the video and
 * the trace at the same time.
 */

const SPARK_W = 320
const SPARK_H = 64
const POLL_MS = 1000

// Mirrors SENSITIVITY_THRESHOLDS in server_py/app/routers/monitor.py. Shown as
// a hint only; the server is authoritative and returns the mapped threshold.
const PRESET_HINT = [8.0, 6.0, 4.5, 3.5, 2.5, 2.0, 1.5, 1.0, 0.7, 0.4]

const fmtGb = bytes => (bytes < 0 ? '—' : `${(bytes / 1024 ** 3).toFixed(1)} GB`)

export default function ClippingPanel({ clippingOn, onToggle, onClipSaved }) {
  const [state, setState]   = useState(null)  // full /api/monitor payload
  const [usage, setUsage]   = useState(null)  // /api/recordings/usage
  const [trace, setTrace]   = useState([])    // recent motion levels
  const [open, setOpen]     = useState(false)
  const saveTimer   = useRef(null)
  const pendingSave = useRef({})   // edits not yet POSTed, merged not replaced
  const settleUntil = useRef(0)    // ignore poll results until this timestamp
  const lastCount   = useRef(null)
  // App passes an inline arrow, so this prop is a new function every render.
  // Held in a ref rather than a dependency so the poll interval isn't torn
  // down and restarted on every parent render.
  const onClipSavedRef = useRef(onClipSaved)
  onClipSavedRef.current = onClipSaved

  // ── Polling ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        const [m, u] = await Promise.all([
          fetch('/api/monitor').then(r => r.json()),
          fetch('/api/recordings/usage').then(r => (r.ok ? r.json() : null)),
        ])
        // A poll issued before an edit can land after it. Applying it would
        // snap a slider back to the old value mid-drag, and the agent would
        // keep running on whatever the poll said. Skip while an edit is
        // unsent or has only just been acknowledged.
        const busy = Date.now() < settleUntil.current || Object.keys(pendingSave.current).length > 0
        if (!busy) setState(m)
        if (u) {
          setUsage(u)
          // Refresh the gallery when a clip actually lands on disk. The motion
          // telemetry's `captured` flag fires when a clip is QUEUED, a segment
          // length before the upload, so refetching on it showed the gallery a
          // file that did not exist yet — and queued clips can still be
          // dropped without ever being saved.
          if (lastCount.current !== null && u.count > lastCount.current) onClipSavedRef.current?.()
          lastCount.current = u.count
        }
      } catch { /* transient — keep showing the last good values */ }
    }
    poll()
    const id = setInterval(poll, 5000)
    return () => clearInterval(id)
  }, [])

  // The motion trace only moves while the agent is clipping (that's the only
  // time it posts telemetry), so don't poll it the rest of the time.
  useEffect(() => {
    if (!clippingOn && !open) return undefined
    const poll = async () => {
      try {
        const d = await fetch('/api/motion').then(r => (r.ok ? r.json() : null))
        if (d) setTrace(d.events.slice(-120))
      } catch { /* ignore */ }
    }
    poll()
    const id = setInterval(poll, POLL_MS)
    return () => clearInterval(id)
  }, [clippingOn, open])

  // ── Saving (debounced — the slider fires continuously while dragging) ─────
  // Edits MERGE into one pending patch rather than replacing it. With a single
  // shared timer and a per-call patch, nudging the slider and then touching
  // cooldown within the debounce window cancelled the slider's POST outright:
  // the optimistic value stayed on screen while the agent never heard about it.
  const save = useCallback((patch, { immediate = false } = {}) => {
    setState(prev => ({ ...prev, ...patch })) // optimistic
    pendingSave.current = { ...pendingSave.current, ...patch }
    clearTimeout(saveTimer.current)

    const send = async () => {
      const body = pendingSave.current
      pendingSave.current = {}
      if (Object.keys(body).length === 0) return
      try {
        const res = await fetch('/api/monitor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (res.ok) {
          const next = await res.json()
          // Only adopt the server's copy if nothing new was typed while this
          // request was in flight; the server owns the mapped threshold.
          if (Object.keys(pendingSave.current).length === 0) setState(next)
        }
      } catch { /* the next poll re-syncs */ }
      settleUntil.current = Date.now() + 1000
    }

    settleUntil.current = Date.now() + 2000
    if (immediate) send()
    else saveTimer.current = setTimeout(send, 400)
  }, [])

  useEffect(() => () => clearTimeout(saveTimer.current), [])

  if (!state) return null

  const sensitivity = state.clippingSensitivity ?? 5
  const threshold   = state.clippingMotionThreshold ?? PRESET_HINT[sensitivity - 1]
  const cooldown    = state.clippingCooldownSec ?? 8
  const dryRun      = !!state.clippingDryRun
  const stopped     = state.clippingStoppedReason

  const maxUnlabeled = usage?.maxUnlabeled ?? 0
  const unlabeled    = usage?.unlabeled ?? 0
  const pct          = maxUnlabeled > 0 ? Math.min(100, (unlabeled / maxUnlabeled) * 100) : 0
  const budgetTone   = pct >= 100 ? 'danger' : pct >= 80 ? 'warn' : 'ok'

  // ── Sparkline ────────────────────────────────────────────────────────────
  // Scale to the taller of the observed peak and the threshold, so the line is
  // always on screen even when the room is completely still.
  const levels  = trace.map(e => e.level)
  const peak    = Math.max(threshold * 1.4, ...levels, 1)
  const x = i => (trace.length < 2 ? 0 : (i / (trace.length - 1)) * SPARK_W)
  const y = v => SPARK_H - (Math.min(v, peak) / peak) * SPARK_H
  const path = trace.length < 2 ? '' : levels.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const thresholdY = y(threshold)
  const latest = levels.length ? levels[levels.length - 1] : null

  return (
    <div className={`clip-panel ${clippingOn ? 'clip-active' : ''}`}>
      <button className="clip-head" onClick={() => setOpen(o => !o)}>
        <span className="clip-title">✂ Clipping Mode</span>
        <span className={`clip-state ${clippingOn ? 'on' : ''}`}>
          {clippingOn ? (dryRun ? 'DRY RUN' : 'HARVESTING') : 'off'}
        </span>
        {maxUnlabeled > 0 && (
          <span className="clip-count">{unlabeled} / {maxUnlabeled} unlabeled</span>
        )}
        <span className="clip-chevron">{open ? '▾' : '▸'}</span>
      </button>

      {stopped && (
        <div className="clip-banner">
          <span>⛔ Stopped automatically — {stopped}</span>
          <button className="clip-rearm" onClick={onToggle}>Resume</button>
        </div>
      )}

      {clippingOn && !stopped && (
        <div className="clip-notice">
          Behavior predictions, email alerts and label suggestions are paused while
          harvesting — the classifier isn't running. A quiet Activity Log is expected.
        </div>
      )}

      {open && (
        <div className="clip-body">
          {/* ── Live motion trace ─────────────────────────────────────── */}
          <div className="clip-field">
            <div className="clip-label">
              <span>Live motion</span>
              <span className="clip-value">
                {latest === null ? 'no data yet' : `${latest.toFixed(1)} now · threshold ${threshold.toFixed(1)}`}
              </span>
            </div>
            <svg className="clip-spark" viewBox={`0 0 ${SPARK_W} ${SPARK_H}`} preserveAspectRatio="none">
              <line x1="0" x2={SPARK_W} y1={thresholdY} y2={thresholdY} className="clip-spark-threshold" />
              {path && <path d={path} className="clip-spark-line" />}
              {trace.map((e, i) => e.wouldCapture && (
                <circle key={i} cx={x(i)} cy={y(e.level)} r="2.5" className="clip-spark-hit" />
              ))}
            </svg>
            <div className="clip-hint">
              {clippingOn
                ? 'Dots mark frames that triggered a clip. Anything above the line is captured.'
                : 'Turn clipping on (or dry run) to see the trace — the agent only reports motion while harvesting.'}
            </div>
          </div>

          {/* ── Sensitivity ───────────────────────────────────────────── */}
          <div className="clip-field">
            <div className="clip-label">
              <span>Sensitivity</span>
              <span className="clip-value">{sensitivity} / 10 → motion ≥ {threshold.toFixed(1)}</span>
            </div>
            <input
              type="range" min="1" max="10" step="1" value={sensitivity}
              onChange={e => {
                const n = Number(e.target.value)
                save({ clippingSensitivity: n, clippingMotionThreshold: PRESET_HINT[n - 1] })
              }}
            />
            <div className="clip-hint">Higher catches more — including more empty footage.</div>
          </div>

          {/* ── Cooldown ──────────────────────────────────────────────── */}
          <div className="clip-field">
            <div className="clip-label">
              <span>Cooldown</span>
              <span className="clip-value">{cooldown}s between clips</span>
            </div>
            <input
              type="range" min="4" max="120" step="1" value={cooldown}
              onChange={e => save({ clippingCooldownSec: Number(e.target.value) })}
            />
            <div className="clip-hint">
              At or above the clip length (8s) you get at most one clip per segment.
            </div>
          </div>

          {/* ── Dry run ───────────────────────────────────────────────── */}
          <label className="clip-check">
            <input
              type="checkbox" checked={dryRun}
              onChange={e => save({ clippingDryRun: e.target.checked }, { immediate: true })}
            />
            <span>
              Dry run — report what <em>would</em> be captured without saving anything.
              Worth ten minutes before leaving it running overnight.
            </span>
          </label>

          {/* ── Budget ────────────────────────────────────────────────── */}
          {usage && (
            <div className="clip-field">
              <div className="clip-label">
                <span>Labeling backlog</span>
                <span className="clip-value">
                  {unlabeled} unlabeled of {usage.count} clips · {fmtGb(usage.freeBytes)} free
                </span>
              </div>
              <div className="clip-bar">
                <div className={`clip-bar-fill ${budgetTone}`} style={{ width: `${pct}%` }} />
              </div>
              <div className="clip-hint">
                Harvesting stops at {maxUnlabeled} unlabeled clips. Labeled clips don't
                count, so working through Label Studio frees room. Nothing is ever deleted.
              </div>
            </div>
          )}
        </div>
      )}

      <style>{`
        .clip-panel {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          margin-top: 16px;
          overflow: hidden;
        }
        .clip-active { border-color: #ffb34766; }

        .clip-head {
          display: flex;
          align-items: center;
          gap: 12px;
          width: 100%;
          padding: 12px 16px;
          background: none;
          border: none;
          cursor: pointer;
          color: var(--text-primary);
          font-family: var(--font-display);
          font-size: 13px;
          text-align: left;
        }
        .clip-head:hover { background: var(--bg-card-hover); }
        .clip-title { flex-shrink: 0; }
        .clip-state {
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.08em;
          color: var(--text-muted);
        }
        .clip-state.on { color: #ffb347; }
        .clip-count {
          margin-left: auto;
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--text-secondary);
        }
        .clip-chevron { color: var(--text-muted); font-size: 10px; }

        .clip-banner {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 10px 16px;
          background: rgba(200, 122, 110, 0.12);
          border-top: 1px solid var(--red-dim);
          color: var(--red);
          font-size: 12px;
        }
        .clip-rearm {
          margin-left: auto;
          background: none;
          border: 1px solid var(--red-dim);
          border-radius: 99px;
          color: var(--red);
          padding: 3px 12px;
          font-size: 11px;
          cursor: pointer;
        }
        .clip-rearm:hover { background: rgba(200, 122, 110, 0.12); }

        .clip-notice {
          padding: 10px 16px;
          border-top: 1px solid var(--border);
          color: var(--text-secondary);
          font-size: 11.5px;
          line-height: 1.5;
        }

        .clip-body {
          padding: 16px;
          border-top: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          gap: 20px;
        }
        .clip-field { display: flex; flex-direction: column; gap: 7px; }
        .clip-label {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 12px;
          font-size: 12px;
          color: var(--text-primary);
        }
        .clip-value {
          font-family: var(--font-mono);
          font-size: 11px;
          color: var(--accent);
        }
        .clip-hint {
          font-size: 11px;
          color: var(--text-muted);
          line-height: 1.45;
        }
        .clip-field input[type="range"] { width: 100%; accent-color: #ffb347; }

        .clip-spark {
          width: 100%;
          height: 64px;
          background: var(--bg-deep);
          border: 1px solid var(--border);
          border-radius: 4px;
        }
        .clip-spark-line {
          fill: none;
          stroke: var(--motion-color);
          stroke-width: 1.5;
          vector-effect: non-scaling-stroke;
        }
        .clip-spark-threshold {
          stroke: #ffb347;
          stroke-width: 1;
          stroke-dasharray: 4 3;
          vector-effect: non-scaling-stroke;
        }
        .clip-spark-hit { fill: #ffb347; }

        .clip-check {
          display: flex;
          gap: 10px;
          align-items: flex-start;
          font-size: 11.5px;
          color: var(--text-secondary);
          line-height: 1.45;
          cursor: pointer;
        }
        .clip-check input { margin-top: 2px; accent-color: #ffb347; }

        .clip-bar {
          height: 6px;
          background: var(--bg-deep);
          border-radius: 99px;
          overflow: hidden;
        }
        .clip-bar-fill { height: 100%; transition: width 0.3s; }
        .clip-bar-fill.ok     { background: var(--green); }
        .clip-bar-fill.warn   { background: #ffb347; }
        .clip-bar-fill.danger { background: var(--red); }
      `}</style>
    </div>
  )
}
