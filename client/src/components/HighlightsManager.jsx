import React, { useState, useEffect, useCallback } from 'react'
import { LABEL_COLOR as BASE_COLOR } from '../labels.js'

const labelDisplay = (l) => (l ? l.charAt(0).toUpperCase() + l.slice(1) : '')
const fmtDate = (iso) =>
  new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

/**
 * HighlightsManager
 * Owner-only tool to curate the public front page. The homepage auto-features
 * every non-normal clip; here the owner can hide any they don't want shown.
 */
export default function HighlightsManager() {
  const [items, setItems]   = useState([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy]     = useState(null) // filename currently toggling

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/highlights')
      const { highlights } = await res.json()
      setItems(highlights || [])
    } catch (err) {
      console.error('Failed to load highlights:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const toggle = async (item) => {
    const next = !item.hidden
    setBusy(item.filename)
    // optimistic
    setItems(prev => prev.map(i => i.filename === item.filename ? { ...i, hidden: next } : i))
    try {
      const res = await fetch(`/api/highlights/${item.filename}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hidden: next }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
    } catch (err) {
      console.error('Toggle failed:', err)
      setItems(prev => prev.map(i => i.filename === item.filename ? { ...i, hidden: !next } : i)) // revert
    } finally {
      setBusy(null)
    }
  }

  const shownCount  = items.filter(i => !i.hidden).length
  const hiddenCount = items.length - shownCount

  return (
    <div className="hl">
      <div className="hl-head">
        <div>
          <div className="hl-title">Homepage Highlights</div>
          <p className="hl-sub">
            These are the clips shown on your public front page
            (<a href="/" target="_blank" rel="noreferrer">bunny-tracker.app</a>).
            Every non-normal clip is featured automatically — hide any you don't
            want the public to see.
          </p>
        </div>
        <div className="hl-stats">
          <span className="hl-stat hl-stat-shown">👁 {shownCount} public</span>
          <span className="hl-stat hl-stat-hidden">🚫 {hiddenCount} hidden</span>
        </div>
      </div>

      {loading && <div className="hl-empty">Loading…</div>}

      {!loading && items.length === 0 && (
        <div className="hl-empty">
          <span style={{ fontSize: 36 }}>🎬</span>
          <p>No highlight clips yet.</p>
          <p className="hl-empty-sub">
            Label some clips as zoomies, grooming, standing, or yawn in Label
            Studio and they'll appear here, ready to feature.
          </p>
        </div>
      )}

      {!loading && items.length > 0 && (
        <div className="hl-grid">
          {items.map(item => {
            const color = BASE_COLOR[item.label] ?? 'var(--text-muted)'
            return (
              <div key={item.filename} className={`hl-card ${item.hidden ? 'is-hidden' : ''}`}>
                <div className="hl-thumb" style={{ borderColor: color + '66' }}>
                  <video muted preload="metadata" src={`/recordings/${item.filename}#t=0.5`} />
                  <span className="hl-badge" style={{ color, borderColor: color + '88' }}>
                    {labelDisplay(item.label)}
                  </span>
                  {item.hidden && <div className="hl-hidden-veil">Hidden</div>}
                </div>
                <div className="hl-meta">
                  <span className="hl-date">{fmtDate(item.createdAt)}</span>
                  <button
                    className={`hl-toggle ${item.hidden ? 'show' : 'hide'}`}
                    onClick={() => toggle(item)}
                    disabled={busy === item.filename}
                  >
                    {item.hidden ? '👁 Show' : '🚫 Hide'}
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <style>{`
        .hl { padding: 20px 28px; display: flex; flex-direction: column; gap: 16px; }

        .hl-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 16px;
          flex-wrap: wrap;
        }
        .hl-title { font-family: var(--font-display); font-size: 18px; color: var(--text-primary); }
        .hl-sub {
          font-size: 12px; color: var(--text-muted); max-width: 560px; margin-top: 4px; line-height: 1.5;
        }
        .hl-sub a { color: var(--accent); text-decoration: none; }
        .hl-sub a:hover { text-decoration: underline; }

        .hl-stats { display: flex; gap: 8px; flex-shrink: 0; }
        .hl-stat {
          font-size: 11px; font-family: var(--font-mono);
          padding: 4px 10px; border-radius: 99px; border: 1px solid var(--border);
          white-space: nowrap;
        }
        .hl-stat-shown  { color: var(--green); border-color: #7ab87a55; }
        .hl-stat-hidden { color: var(--text-muted); }

        .hl-empty {
          display: flex; flex-direction: column; align-items: center; justify-content: center;
          gap: 8px; padding: 60px 20px; color: var(--text-muted); font-size: 13px; text-align: center;
        }
        .hl-empty-sub { font-size: 11px; max-width: 360px; }

        .hl-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 14px;
        }

        .hl-card { display: flex; flex-direction: column; gap: 6px; transition: opacity 0.15s; }
        .hl-card.is-hidden { opacity: 0.55; }

        .hl-thumb {
          position: relative;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          overflow: hidden;
        }
        .hl-thumb video {
          width: 100%; aspect-ratio: 16 / 9; object-fit: cover; display: block;
        }

        .hl-badge {
          position: absolute; top: 6px; left: 6px;
          font-size: 9px; font-family: var(--font-display);
          letter-spacing: 0.05em; text-transform: uppercase;
          background: rgba(10,10,10,0.78); border: 1px solid; border-radius: 3px;
          padding: 1px 6px;
        }

        .hl-hidden-veil {
          position: absolute; inset: 0;
          display: flex; align-items: center; justify-content: center;
          background: rgba(10,8,6,0.55);
          color: var(--text-secondary);
          font-family: var(--font-display); font-size: 13px; letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .hl-meta { display: flex; align-items: center; justify-content: space-between; gap: 6px; }
        .hl-date { font-size: 10px; color: var(--text-muted); }

        .hl-toggle {
          font-size: 11px; font-family: var(--font-mono);
          padding: 4px 10px; border-radius: var(--radius);
          border: 1px solid var(--border); background: var(--bg-card);
          color: var(--text-secondary); cursor: pointer; transition: all 0.15s;
          white-space: nowrap;
        }
        .hl-toggle:disabled { opacity: 0.5; cursor: default; }
        .hl-toggle.hide:hover { color: var(--red); border-color: var(--red-dim); }
        .hl-toggle.show:hover { color: var(--green); border-color: #7ab87a55; }

        @media (max-width: 640px) {
          .hl { padding: 16px 14px; }
          .hl-grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
        }
      `}</style>
    </div>
  )
}
