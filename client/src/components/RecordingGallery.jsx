import React, { useState, useEffect, useCallback } from 'react'
import { LABEL_COLOR as BASE_COLOR } from '../labels.js'

const labelColor = (lbl) => {
  if (!lbl) return 'var(--text-muted)'
  return BASE_COLOR[lbl] ?? 'var(--text-muted)'
}

const labelDisplay = (lbl) => {
  if (!lbl) return 'Unlabeled'
  return lbl.charAt(0).toUpperCase() + lbl.slice(1)
}

const FILTER_OPTIONS = [
  { value: 'all',          label: 'All' },
  { value: 'zoomies',      label: 'Zoomies' },
  { value: 'yawn',         label: 'Yawn' },
  { value: 'grooming',     label: 'Grooming' },
  { value: 'standing',     label: 'Standing' },
  { value: 'normal',       label: 'Normal' },
  { value: 'unlabeled',    label: 'Unlabeled' },
]

export default function RecordingGallery({ refreshTrigger }) {
  const [recordings, setRecordings] = useState([])
  const [labels, setLabels]         = useState({})
  const [loading, setLoading]       = useState(false)
  const [filter, setFilter]         = useState('all')

  const fetchAll = useCallback(async () => {
    setLoading(true)
    try {
      const [recRes, lblRes] = await Promise.all([
        fetch('/api/recordings'),
        fetch('/api/labels'),
      ])
      const { recordings: recs } = await recRes.json()
      const { labels: lbls }     = await lblRes.json()
      setRecordings((recs || []).sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)))
      setLabels(lbls || {})
    } catch (err) {
      console.error('Failed to load recordings:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll, refreshTrigger])

  const deleteRecording = async (filename) => {
    try {
      await fetch(`/api/recordings/${filename}`, { method: 'DELETE' })
      setRecordings(prev => prev.filter(r => r.filename !== filename))
    } catch (err) {
      console.error('Failed to delete recording:', err)
    }
  }

  const formatDate = (isoStr) => {
    const d = new Date(isoStr)
    return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  const filtered = recordings.filter(r => {
    if (filter === 'all') return true
    const lbl = labels[r.filename] ?? null
    if (filter === 'unlabeled') return lbl === null
    return lbl === filter
  })

  return (
    <div className="recordings">
      <div className="recordings-header">
        <span className="recordings-title">Recordings</span>
        <div className="recordings-header-right">
          <select
            className="filter-select"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          >
            {FILTER_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <span className="recordings-count">{filtered.length} / {recordings.length}</span>
        </div>
      </div>

      {loading && <div className="recordings-loading">Loading…</div>}

      {!loading && recordings.length === 0 && (
        <div className="recordings-empty">
          <span>🎥</span>
          <span>No recordings yet. Enable auto record or hit record.</span>
        </div>
      )}

      {!loading && recordings.length > 0 && filtered.length === 0 && (
        <div className="recordings-empty">
          <span>🔍</span>
          <span>No {filter} clips yet.</span>
        </div>
      )}

      <div className="recordings-grid">
        {filtered.map(recording => {
          const lbl = labels[recording.filename] ?? null
          const color = labelColor(lbl)
          return (
            <div key={recording.filename} className="recording-item">
              <div className="recording-thumb" style={{ borderColor: lbl ? color + '66' : 'var(--border)' }}>
                <video controls muted preload="none" src={`/recordings/${recording.filename}`} />
                {lbl && (
                  <div className="recording-label-badge" style={{ color, borderColor: color + '88' }}>
                    {labelDisplay(lbl)}
                  </div>
                )}
              </div>
              <div className="recording-meta">
                <span className="recording-time">{formatDate(recording.createdAt)}</span>
                <button className="delete-btn" onClick={() => deleteRecording(recording.filename)} title="Delete">✕</button>
              </div>
            </div>
          )
        })}
      </div>

      <style>{`
        .recordings {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          overflow: hidden;
        }

        .recordings-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 14px;
          border-bottom: 1px solid var(--border);
          background: var(--bg-surface);
          gap: 8px;
        }

        .recordings-title {
          font-family: var(--font-display);
          font-size: 13px;
          color: var(--text-secondary);
        }

        .recordings-header-right {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .filter-select {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          color: var(--text-secondary);
          font-size: 11px;
          padding: 3px 6px;
          cursor: pointer;
          outline: none;
        }
        .filter-select:focus { border-color: var(--accent); }

        .recordings-count {
          color: var(--text-muted);
          font-size: 11px;
          white-space: nowrap;
        }

        .recordings-loading,
        .recordings-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 32px;
          color: var(--text-muted);
          font-size: 12px;
        }

        .recordings-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
          gap: 8px;
          padding: 12px;
          max-height: 380px;
          overflow-y: auto;
        }

        .recording-item {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .recording-thumb {
          position: relative;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          overflow: hidden;
          transition: border-color 0.15s;
        }

        .recording-thumb video {
          width: 100%;
          aspect-ratio: 16 / 9;
          display: block;
          object-fit: cover;
        }

        .recording-label-badge {
          position: absolute;
          bottom: 4px;
          left: 4px;
          font-size: 8px;
          font-family: var(--font-display);
          letter-spacing: 0.05em;
          text-transform: uppercase;
          background: rgba(10,10,10,0.75);
          border: 1px solid;
          border-radius: 3px;
          padding: 1px 5px;
          pointer-events: none;
        }

        .recording-meta {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 4px;
        }

        .recording-time {
          color: var(--text-muted);
          font-size: 9px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .delete-btn {
          background: none;
          color: var(--text-muted);
          font-size: 10px;
          padding: 1px 3px;
          border-radius: 2px;
          flex-shrink: 0;
          transition: color 0.15s;
        }
        .delete-btn:hover { color: var(--red); }
      `}</style>
    </div>
  )
}
