import React, { useState, useEffect, useCallback, useRef } from 'react'

const LABEL_ZOOMIES  = 'zoomies'
const LABEL_YAWN     = 'yawn'
const LABEL_NORMAL   = 'normal'
const LABEL_GROOMING = 'grooming'
const LABEL_STANDING = 'standing'
const ACCEPTED_EXTENSIONS = '.mp4,.mov,.avi,.mkv,.webm,.m4v'

// ── Import Panel ────────────────────────────────────────────────────────────
function ImportPanel({ onImportDone }) {
  const [dragging,  setDragging]  = useState(false)
  const [files,     setFiles]     = useState([])   // [{ file, status, detail }]
  const [uploading, setUploading] = useState(false)
  const [open,      setOpen]      = useState(false)
  const inputRef = useRef(null)

  const addFiles = (incoming) => {
    const videoFiles = Array.from(incoming).filter(f =>
      f.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name)
    )
    setFiles(prev => {
      const existingNames = new Set(prev.map(e => e.file.name))
      const newEntries = videoFiles
        .filter(f => !existingNames.has(f.name))
        .map(f => ({ file: f, status: 'pending', detail: '' }))
      return [...prev, ...newEntries]
    })
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }

  const removeFile = (name) =>
    setFiles(prev => prev.filter(e => e.file.name !== name))

  const upload = async () => {
    if (uploading || files.length === 0) return
    setUploading(true)

    // Mark all pending as uploading
    setFiles(prev => prev.map(e =>
      e.status === 'pending' ? { ...e, status: 'converting' } : e
    ))

    const formData = new FormData()
    files.filter(e => e.status === 'converting').forEach(e =>
      formData.append('videos', e.file)
    )

    try {
      const res  = await fetch('/api/import', { method: 'POST', body: formData })
      const data = await res.json()

      setFiles(prev => prev.map(e => {
        const result = data.results?.find(r => r.original === e.file.name)
        if (!result) return e
        return { ...e, status: result.status, detail: result.detail || '' }
      }))

      onImportDone()
    } catch (err) {
      setFiles(prev => prev.map(e =>
        e.status === 'converting' ? { ...e, status: 'error', detail: err.message } : e
      ))
    } finally {
      setUploading(false)
    }
  }

  const clearDone = () => setFiles(prev => prev.filter(e => e.status !== 'ok'))

  const pendingCount = files.filter(e => e.status === 'pending').length
  const doneCount    = files.filter(e => e.status === 'ok').length

  return (
    <div className="import-panel">
      <button className="import-toggle" onClick={() => setOpen(o => !o)}>
        <span>{open ? '▾' : '▸'} Import Footage</span>
        <span className="import-toggle-sub">mp4, mov, avi, mkv, webm</span>
      </button>

      {open && (
        <div className="import-body">
          {/* Drop zone */}
          <div
            className={`drop-zone ${dragging ? 'drop-zone-active' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <span className="drop-icon">📂</span>
            <p className="drop-text">Drop video files here or click to browse</p>
            <p className="drop-sub">Up to 20 files · 500 MB each · ffmpeg converts automatically</p>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={ACCEPTED_EXTENSIONS}
              style={{ display: 'none' }}
              onChange={e => addFiles(e.target.files)}
            />
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="import-file-list">
              {files.map(({ file, status, detail }) => (
                <div key={file.name} className={`import-file-row status-${status}`}>
                  <span className="import-file-icon">
                    {status === 'pending'    && '⏳'}
                    {status === 'converting' && '⚙️'}
                    {status === 'ok'         && '✓'}
                    {status === 'error'      && '✕'}
                  </span>
                  <span className="import-file-name" title={file.name}>{file.name}</span>
                  <span className="import-file-size">{formatSize(file.size)}</span>
                  {status === 'error' && <span className="import-file-err" title={detail}>failed</span>}
                  {status === 'pending' && (
                    <button className="import-file-remove" onClick={() => removeFile(file.name)}>✕</button>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Actions */}
          {files.length > 0 && (
            <div className="import-actions">
              {doneCount > 0 && (
                <button className="import-clear-btn" onClick={clearDone}>
                  Clear {doneCount} done
                </button>
              )}
              <button
                className="import-upload-btn"
                onClick={upload}
                disabled={uploading || pendingCount === 0}
              >
                {uploading
                  ? '⚙️ Converting…'
                  : `↑ Import ${pendingCount} clip${pendingCount !== 1 ? 's' : ''}`}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function formatDate(isoStr) {
  return new Date(isoStr).toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function LabelingStudio() {
  const [recordings, setRecordings] = useState([])  // [{ filename, createdAt, size }]
  const [labels, setLabels]         = useState({})   // { filename: 'zoomies'|'yawn'|'normal'|'grooming'|'standing' }
  const [suggestions, setSuggestions] = useState({}) // agent-predicted labels awaiting review (not training data)
  const [index, setIndex]           = useState(0)
  const [filter, setFilter]         = useState('all') // 'all' | 'unlabeled' | 'zoomies' | 'yawn' | 'normal' | 'grooming' | 'standing'
  const [saving, setSaving]         = useState(false)
  const [loading, setLoading]       = useState(true)
  const [openMenu, setOpenMenu]     = useState(null)  // filename of open three-dot menu
  const videoRef = useRef(null)

  // ── Load recordings + labels ──────────────────────────────────────────────
  const loadData = useCallback(async () => {
    try {
      const [recRes, labRes] = await Promise.all([
        fetch('/api/recordings'),
        fetch('/api/labels'),
      ])
      const recData = await recRes.json()
      const labData = await labRes.json()
      setRecordings(recData.recordings || [])
      setLabels(labData.labels || {})
      setSuggestions(labData.suggestions || {})
    } catch (err) {
      console.error('Failed to load labeling data:', err)
    }
  }, [])

  useEffect(() => {
    async function load() {
      setLoading(true)
      try {
        const [recRes, labRes] = await Promise.all([
          fetch('/api/recordings'),
          fetch('/api/labels'),
        ])
        const recData = await recRes.json()
        const labData = await labRes.json()
        setRecordings(recData.recordings || [])
        setLabels(labData.labels || {})
        setSuggestions(labData.suggestions || {})
      } catch (err) {
        console.error('Failed to load labeling data:', err)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  // ── Filtered + sorted clip list ───────────────────────────────────────────
  const filtered = recordings.filter(r => {
    if (filter === 'unlabeled') return !labels[r.filename]
    if (filter === 'zoomies')     return labels[r.filename] === LABEL_ZOOMIES
    if (filter === 'yawn')      return labels[r.filename] === LABEL_YAWN
    if (filter === 'normal')    return labels[r.filename] === LABEL_NORMAL
    if (filter === 'grooming')  return labels[r.filename] === LABEL_GROOMING
    if (filter === 'standing')  return labels[r.filename] === LABEL_STANDING
    return true
  })

  const current = filtered[index] || null

  // Reset index when filter changes
  useEffect(() => { setIndex(0) }, [filter])

  // Replay video when clip changes
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load()
      videoRef.current.play().catch(() => {})
    }
  }, [current?.filename])

  // ── Label actions ─────────────────────────────────────────────────────────
  const applyLabel = useCallback(async (label) => {
    if (!current || saving) return
    setSaving(true)
    try {
      const res = await fetch(`/api/labels/${current.filename}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      })
      if (!res.ok) {
        const detail = res.status === 401
          ? 'Not logged in — your session expired. Reload and log in again.'
          : `Save failed (HTTP ${res.status})`
        alert(detail)
        return // do NOT optimistically update — the label did not save
      }
      setLabels(prev => ({ ...prev, [current.filename]: label }))
      // Labeling resolves the agent's suggestion (server clears it too)
      setSuggestions(prev => {
        const next = { ...prev }
        delete next[current.filename]
        return next
      })
      // Advance to next clip automatically
      setIndex(prev => Math.min(prev + 1, filtered.length - 1))
    } catch (err) {
      console.error('Label save failed:', err)
      alert('Label save failed — check your connection.')
    } finally {
      setSaving(false)
    }
  }, [current, saving, filtered.length])

  const removeLabel = useCallback(async () => {
    if (!current || !labels[current.filename]) return
    try {
      await fetch(`/api/labels/${current.filename}`, { method: 'DELETE' })
      setLabels(prev => {
        const next = { ...prev }
        delete next[current.filename]
        return next
      })
    } catch (err) {
      console.error('Label remove failed:', err)
    }
  }, [current, labels])

  const deleteClip = useCallback(async (filename) => {
    const target = filename || current?.filename
    if (!target) return
    if (!window.confirm(`Delete "${target}" permanently? This cannot be undone.`)) return
    setOpenMenu(null)
    try {
      await fetch(`/api/recordings/${target}`, { method: 'DELETE' })
      await fetch(`/api/labels/${target}`, { method: 'DELETE' }).catch(() => {})
      setLabels(prev => {
        const next = { ...prev }
        delete next[target]
        return next
      })
      setRecordings(prev => prev.filter(r => r.filename !== target))
      setIndex(prev => Math.max(0, prev > 0 ? prev - 1 : 0))
    } catch (err) {
      console.error('Delete failed:', err)
    }
  }, [current])

  const downloadClip = useCallback((filename) => {
    const target = filename || current?.filename
    if (!target) return
    setOpenMenu(null)
    const a = document.createElement('a')
    a.href = `/recordings/${target}`
    a.download = target
    a.click()
  }, [current])

  // ── Close three-dot menu on outside click ────────────────────────────────
  useEffect(() => {
    if (!openMenu) return
    const close = () => setOpenMenu(null)
    window.addEventListener('click', close)
    return () => window.removeEventListener('click', close)
  }, [openMenu])

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return
      if (e.key === 'z' || e.key === 'Z') applyLabel(LABEL_ZOOMIES)
      if (e.key === 'y' || e.key === 'Y') applyLabel(LABEL_YAWN)
      if (e.key === 'n' || e.key === 'N') applyLabel(LABEL_NORMAL)
      if (e.key === 'g' || e.key === 'G') applyLabel(LABEL_GROOMING)
      if (e.key === 's' || e.key === 'S') applyLabel(LABEL_STANDING)
      if (e.key === 'ArrowRight' || e.key === 'd')
        setIndex(prev => Math.min(prev + 1, filtered.length - 1))
      if (e.key === 'ArrowLeft' || e.key === 'a')
        setIndex(prev => Math.max(prev - 1, 0))
      if (e.key === 'Delete' || e.key === 'Backspace') removeLabel()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [applyLabel, removeLabel, filtered.length])

  // ── Export labels.json ────────────────────────────────────────────────────
  const exportLabels = () => {
    const blob = new Blob([JSON.stringify(labels, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = 'labels.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  // ── Stats ─────────────────────────────────────────────────────────────────
  const totalCount    = recordings.length
  const zoomiesCount  = Object.values(labels).filter(l => l === LABEL_ZOOMIES).length
  const yawnCount     = Object.values(labels).filter(l => l === LABEL_YAWN).length
  const normalCount   = Object.values(labels).filter(l => l === LABEL_NORMAL).length
  const groomingCount = Object.values(labels).filter(l => l === LABEL_GROOMING).length
  const standingCount = Object.values(labels).filter(l => l === LABEL_STANDING).length
  const labeledCount  = zoomiesCount + yawnCount + normalCount + groomingCount + standingCount
  const pctDone      = totalCount > 0 ? Math.round((labeledCount / totalCount) * 100) : 0

  if (loading) {
    return (
      <div className="studio-loading">
        <span>🐇</span>
        <p>Loading clips…</p>
        <style>{`.studio-loading { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:12px; min-height:400px; color:var(--text-muted); font-size:14px; }`}</style>
      </div>
    )
  }

  if (recordings.length === 0) {
    return (
      <div className="studio-empty">
        <span>🎥</span>
        <p>No recordings to label yet.</p>
        <p className="sub">Record clips from the Camera tab, or import footage below.</p>
        <ImportPanel onImportDone={loadData} />
        <style>{`
          .studio-empty { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; min-height:400px; color:var(--text-muted); }
          .studio-empty span { font-size:48px; }
          .studio-empty p { font-size:14px; }
          .studio-empty .sub { font-size:12px; }
        `}</style>
      </div>
    )
  }

  const currentLabel      = current ? labels[current.filename] : null
  const currentSuggestion = current && !currentLabel ? suggestions[current.filename] : null

  return (
    <div className="studio">

      <ImportPanel onImportDone={loadData} />

      {/* ── Progress header ─────────────────────────────────────────────── */}
      <div className="studio-header">
        <div className="studio-title">Label Studio</div>
        <div className="studio-stats">
          <span className="stat-chip stat-zoomies">{zoomiesCount} zoomies</span>
          <span className="stat-chip stat-yawn">{yawnCount} yawn</span>
          <span className="stat-chip stat-normal">{normalCount} normal</span>
          <span className="stat-chip stat-grooming">{groomingCount} grooming</span>
          <span className="stat-chip stat-standing">{standingCount} standing</span>
          <span className="stat-chip stat-unlabeled">{totalCount - labeledCount} unlabeled</span>
        </div>
        <button className="export-btn" onClick={exportLabels} disabled={labeledCount === 0}>
          ↓ Export labels.json
        </button>
      </div>

      {/* ── Progress bar ────────────────────────────────────────────────── */}
      <div className="progress-track" title={`${labeledCount} of ${totalCount} labeled`}>
        <div
          className="progress-zoomies"
          style={{ width: `${totalCount > 0 ? (zoomiesCount / totalCount) * 100 : 0}%` }}
        />
        <div
          className="progress-yawn"
          style={{ width: `${totalCount > 0 ? (yawnCount / totalCount) * 100 : 0}%` }}
        />
        <div
          className="progress-normal"
          style={{ width: `${totalCount > 0 ? (normalCount / totalCount) * 100 : 0}%` }}
        />
        <div
          className="progress-grooming"
          style={{ width: `${totalCount > 0 ? (groomingCount / totalCount) * 100 : 0}%` }}
        />
        <div
          className="progress-standing"
          style={{ width: `${totalCount > 0 ? (standingCount / totalCount) * 100 : 0}%` }}
        />
      </div>
      <div className="progress-label">{pctDone}% labeled — {labeledCount} / {totalCount} clips</div>

      {/* ── Filter tabs ─────────────────────────────────────────────────── */}
      <div className="filter-tabs">
        {['all', 'unlabeled', 'zoomies', 'yawn', 'normal', 'grooming', 'standing'].map(f => (
          <button
            key={f}
            className={`filter-tab ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all'       ? `All (${totalCount})`                      : ''}
            {f === 'unlabeled' ? `Unlabeled (${totalCount - labeledCount})` : ''}
            {f === 'zoomies'   ? `Zoomies (${zoomiesCount})`                : ''}
            {f === 'yawn'      ? `Yawn (${yawnCount})`                      : ''}
            {f === 'normal'    ? `Normal (${normalCount})`                  : ''}
            {f === 'grooming'  ? `Grooming (${groomingCount})`              : ''}
            {f === 'standing'  ? `Standing (${standingCount})`              : ''}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="no-clips">No clips match this filter.</div>
      ) : (
        <div className="studio-body">

          {/* ── Video player ──────────────────────────────────────────────── */}
          <div className="player-section">
            <div className={`player-wrap ${currentLabel === LABEL_ZOOMIES ? 'border-zoomies' : currentLabel === LABEL_YAWN ? 'border-yawn' : currentLabel === LABEL_NORMAL ? 'border-normal' : currentLabel === LABEL_GROOMING ? 'border-grooming' : currentLabel === LABEL_STANDING ? 'border-standing' : ''}`}>
              {current && (
                <video
                  ref={videoRef}
                  key={current.filename}
                  controls
                  loop
                  autoPlay
                  muted
                  className="player-video"
                  src={`/recordings/${current.filename}`}
                />
              )}
              {currentLabel && (
                <div className={`current-label-badge ${currentLabel === LABEL_ZOOMIES ? 'badge-zoomies' : currentLabel === LABEL_YAWN ? 'badge-yawn' : currentLabel === LABEL_GROOMING ? 'badge-grooming' : currentLabel === LABEL_STANDING ? 'badge-standing' : 'badge-normal'}`}>
                  {currentLabel === LABEL_ZOOMIES ? '⚡ ZOOMIES' : currentLabel === LABEL_YAWN ? '🥱 YAWN' : currentLabel === LABEL_GROOMING ? '🐾 GROOMING' : currentLabel === LABEL_STANDING ? '🦘 STANDING' : '🚶 NORMAL'}
                </div>
              )}
            </div>

            {/* Clip metadata */}
            {current && (
              <div className="clip-meta">
                <span className="clip-name">{current.filename}</span>
                <span className="clip-detail">{formatDate(current.createdAt)} · {formatSize(current.size)}</span>
              </div>
            )}

            {/* Agent suggestion — a prediction awaiting review, not a saved label */}
            {currentSuggestion && (
              <div className="suggestion-hint">
                🤖 Agent suggests <strong>{currentSuggestion}</strong> — label to confirm or correct
              </div>
            )}

            {/* Navigation */}
            <div className="nav-row">
              <button
                className="nav-btn"
                onClick={() => setIndex(prev => Math.max(prev - 1, 0))}
                disabled={index === 0}
              >
                ← Prev
              </button>
              <span className="nav-counter">{index + 1} / {filtered.length}</span>
              <button
                className="nav-btn"
                onClick={() => setIndex(prev => Math.min(prev + 1, filtered.length - 1))}
                disabled={index === filtered.length - 1}
              >
                Next →
              </button>
            </div>

            {/* Label buttons */}
            <div className="label-buttons">
              <button
                className={`label-btn btn-zoomies ${currentLabel === LABEL_ZOOMIES ? 'selected' : ''}`}
                onClick={() => applyLabel(LABEL_ZOOMIES)}
                disabled={saving}
              >
                <span className="label-btn-icon">⚡</span>
                <span className="label-btn-text">Zoomies</span>
                <span className="label-btn-key">Z</span>
              </button>

              <button
                className={`label-btn btn-yawn ${currentLabel === LABEL_YAWN ? 'selected' : ''}`}
                onClick={() => applyLabel(LABEL_YAWN)}
                disabled={saving}
              >
                <span className="label-btn-icon">🥱</span>
                <span className="label-btn-text">Yawn</span>
                <span className="label-btn-key">Y</span>
              </button>

              <button
                className={`label-btn btn-normal ${currentLabel === LABEL_NORMAL ? 'selected' : ''}`}
                onClick={() => applyLabel(LABEL_NORMAL)}
                disabled={saving}
              >
                <span className="label-btn-icon">🚶</span>
                <span className="label-btn-text">Normal</span>
                <span className="label-btn-key">N</span>
              </button>

              <button
                className={`label-btn btn-grooming ${currentLabel === LABEL_GROOMING ? 'selected' : ''}`}
                onClick={() => applyLabel(LABEL_GROOMING)}
                disabled={saving}
              >
                <span className="label-btn-icon">🐾</span>
                <span className="label-btn-text">Grooming</span>
                <span className="label-btn-key">G</span>
              </button>

              <button
                className={`label-btn btn-standing ${currentLabel === LABEL_STANDING ? 'selected' : ''}`}
                onClick={() => applyLabel(LABEL_STANDING)}
                disabled={saving}
              >
                <span className="label-btn-icon">🦘</span>
                <span className="label-btn-text">Standing</span>
                <span className="label-btn-key">S</span>
              </button>
            </div>

            <div className="clip-actions">
              {currentLabel && (
                <button className="remove-label-btn" onClick={removeLabel}>
                  ✕ Remove label
                </button>
              )}
              <button className="delete-clip-btn" onClick={() => deleteClip()} title="Permanently delete this clip">
                🗑 Delete clip
              </button>
            </div>

            <div className="shortcut-hint">
              Arrow keys to navigate · Z / Y / N / G / S to label · Delete to clear
            </div>
          </div>

          {/* ── Filmstrip ─────────────────────────────────────────────────── */}
          <div className="filmstrip">
            {filtered.map((r, i) => {
              const lbl = labels[r.filename]
              const menuOpen = openMenu === r.filename
              return (
                <div
                  key={r.filename}
                  className={`strip-thumb ${i === index ? 'strip-current' : ''} ${lbl === LABEL_ZOOMIES ? 'strip-zoomies' : lbl === LABEL_YAWN ? 'strip-yawn' : lbl === LABEL_NORMAL ? 'strip-normal' : lbl === LABEL_GROOMING ? 'strip-grooming' : lbl === LABEL_STANDING ? 'strip-standing' : 'strip-unlabeled'}`}
                  title={`${r.filename} — ${lbl || 'unlabeled'}`}
                >
                  {/* Thumbnail — clicking selects the clip */}
                  <video
                    muted
                    preload="metadata"
                    src={`/recordings/${r.filename}#t=0.5`}
                    className="strip-video"
                    onClick={() => setIndex(i)}
                  />

                  {/* Label badge */}
                  {lbl && (
                    <span className={`strip-badge ${lbl === LABEL_ZOOMIES ? 'strip-badge-zoomies' : lbl === LABEL_YAWN ? 'strip-badge-yawn' : lbl === LABEL_GROOMING ? 'strip-badge-grooming' : lbl === LABEL_STANDING ? 'strip-badge-standing' : 'strip-badge-normal'}`}>
                      {lbl === LABEL_ZOOMIES ? 'Z' : lbl === LABEL_YAWN ? 'Y' : lbl === LABEL_GROOMING ? 'G' : lbl === LABEL_STANDING ? 'S' : 'N'}
                    </span>
                  )}

                  {/* Three-dot menu button */}
                  <button
                    className="strip-menu-btn"
                    onClick={e => { e.stopPropagation(); setOpenMenu(menuOpen ? null : r.filename) }}
                    title="Options"
                  >
                    ⋯
                  </button>

                  {/* Dropdown */}
                  {menuOpen && (
                    <div className="strip-menu" onClick={e => e.stopPropagation()}>
                      <button className="strip-menu-item" onClick={() => downloadClip(r.filename)}>
                        ↓ Download
                      </button>
                      <button className="strip-menu-item strip-menu-delete" onClick={() => deleteClip(r.filename)}>
                        🗑 Delete
                      </button>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

        </div>
      )}

      <style>{`
        .studio {
          display: flex;
          flex-direction: column;
          gap: 12px;
          padding: 20px 28px;
        }

        /* Header */
        .studio-header {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .studio-title {
          font-family: var(--font-display);
          font-size: 18px;
          color: var(--text-primary);
          margin-right: auto;
        }

        .studio-stats {
          display: flex;
          gap: 6px;
        }

        .stat-chip {
          font-size: 11px;
          padding: 3px 10px;
          border-radius: 20px;
          border: 1px solid transparent;
        }

        .stat-zoomies     { background: rgba(125, 255, 125, 0.1); border-color: rgba(125, 255, 125, 0.3); color: #7dff7d; }
        .stat-yawn      { background: rgba(255, 210, 100, 0.1); border-color: rgba(255, 210, 100, 0.3); color: #ffd264; }
        .stat-normal    { background: rgba(100, 160, 255, 0.1); border-color: rgba(100, 160, 255, 0.3); color: #88aaff; }
        .stat-grooming  { background: rgba(220, 130, 255, 0.1); border-color: rgba(220, 130, 255, 0.3); color: #dc82ff; }
        .stat-standing  { background: rgba(255, 160,  60, 0.1); border-color: rgba(255, 160,  60, 0.3); color: #ff9f3c; }
        .stat-unlabeled { background: var(--bg-card); border-color: var(--border); color: var(--text-muted); }

        .export-btn {
          font-size: 11px;
          padding: 5px 12px;
          border-radius: var(--radius);
          background: var(--bg-card);
          border: 1px solid var(--border);
          color: var(--text-secondary);
          font-family: var(--font-mono);
          transition: all 0.15s;
        }

        .export-btn:hover:not(:disabled) {
          border-color: var(--accent-soft);
          color: var(--accent);
        }

        .export-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Progress */
        .progress-track {
          height: 5px;
          background: var(--bg-card);
          border-radius: 3px;
          overflow: hidden;
          display: flex;
        }

        .progress-zoomies    { height: 100%; background: #7dff7d; transition: width 0.3s ease; }
        .progress-yawn     { height: 100%; background: #ffd264; transition: width 0.3s ease; }
        .progress-normal   { height: 100%; background: #88aaff; transition: width 0.3s ease; }
        .progress-grooming { height: 100%; background: #dc82ff; transition: width 0.3s ease; }
        .progress-standing { height: 100%; background: #ff9f3c; transition: width 0.3s ease; }

        .progress-label {
          font-size: 11px;
          color: var(--text-muted);
          text-align: right;
        }

        /* Filter tabs */
        .filter-tabs {
          display: flex;
          gap: 4px;
          border-bottom: 1px solid var(--border);
          padding-bottom: 0;
        }

        .filter-tab {
          font-size: 11px;
          padding: 6px 14px;
          border-radius: var(--radius) var(--radius) 0 0;
          background: transparent;
          border: 1px solid transparent;
          border-bottom: none;
          color: var(--text-muted);
          font-family: var(--font-mono);
          letter-spacing: 0.04em;
          cursor: pointer;
          transition: all 0.15s;
          margin-bottom: -1px;
        }

        .filter-tab:hover { color: var(--text-secondary); }

        .filter-tab.active {
          background: var(--bg-card);
          border-color: var(--border);
          color: var(--text-primary);
        }

        .no-clips {
          text-align: center;
          color: var(--text-muted);
          font-size: 13px;
          padding: 60px;
        }

        /* Body layout */
        .studio-body {
          display: grid;
          grid-template-columns: 1fr 180px;
          gap: 16px;
          align-items: start;
        }

        /* Player */
        .player-section {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .player-wrap {
          position: relative;
          width: 100%;
          aspect-ratio: 16 / 9;
          background: var(--bg-deep);
          border-radius: var(--radius-lg);
          overflow: hidden;
          border: 2px solid var(--border);
          transition: border-color 0.2s ease;
        }

        .border-zoomies    { border-color: rgba(125, 255, 125, 0.6); }
        .border-yawn     { border-color: rgba(255, 210, 100, 0.6); }
        .border-normal   { border-color: rgba(136, 170, 255, 0.6); }
        .border-grooming { border-color: rgba(220, 130, 255, 0.6); }
        .border-standing { border-color: rgba(255, 160,  60, 0.6); }

        .player-video {
          width: 100%;
          height: 100%;
          object-fit: contain;
          display: block;
        }

        .current-label-badge {
          position: absolute;
          top: 10px;
          right: 10px;
          font-size: 11px;
          font-family: var(--font-mono);
          letter-spacing: 0.1em;
          padding: 3px 10px;
          border-radius: 4px;
          font-weight: 600;
        }

        .badge-zoomies    { background: rgba(0, 0, 0, 0.7); color: #7dff7d; border: 1px solid rgba(125,255,125,0.4); }
        .badge-yawn     { background: rgba(0, 0, 0, 0.7); color: #ffd264; border: 1px solid rgba(255,210,100,0.4); }
        .badge-normal   { background: rgba(0, 0, 0, 0.7); color: #88aaff; border: 1px solid rgba(136,170,255,0.4); }
        .badge-grooming { background: rgba(0, 0, 0, 0.7); color: #dc82ff; border: 1px solid rgba(220,130,255,0.4); }
        .badge-standing { background: rgba(0, 0, 0, 0.7); color: #ff9f3c; border: 1px solid rgba(255,160, 60,0.4); }

        /* Clip meta */
        .clip-meta {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .clip-name {
          font-size: 10px;
          color: var(--text-muted);
          font-family: var(--font-mono);
          word-break: break-all;
        }

        .clip-detail {
          font-size: 10px;
          color: var(--text-muted);
        }

        /* Navigation */
        .nav-row {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
        }

        .nav-btn {
          font-size: 12px;
          padding: 6px 16px;
          border-radius: var(--radius);
          background: var(--bg-card);
          border: 1px solid var(--border);
          color: var(--text-secondary);
          font-family: var(--font-mono);
          transition: all 0.15s;
        }

        .nav-btn:hover:not(:disabled) {
          background: var(--bg-card-hover);
          color: var(--text-primary);
        }

        .nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }

        .nav-counter {
          font-size: 12px;
          color: var(--text-muted);
          font-family: var(--font-mono);
          min-width: 60px;
          text-align: center;
        }

        /* Label buttons */
        .label-buttons {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
        }

        .label-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 16px;
          border-radius: var(--radius-lg);
          border: 2px solid transparent;
          font-family: var(--font-mono);
          font-size: 14px;
          font-weight: 600;
          letter-spacing: 0.06em;
          cursor: pointer;
          transition: all 0.15s ease;
          position: relative;
        }

        .label-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .btn-zoomies {
          background: rgba(125, 255, 125, 0.08);
          border-color: rgba(125, 255, 125, 0.25);
          color: rgba(125, 255, 125, 0.8);
        }

        .btn-zoomies:hover:not(:disabled),
        .btn-zoomies.selected {
          background: rgba(125, 255, 125, 0.18);
          border-color: rgba(125, 255, 125, 0.7);
          color: #7dff7d;
          box-shadow: 0 0 16px rgba(125, 255, 125, 0.15);
        }

        .btn-yawn {
          background: rgba(255, 210, 100, 0.08);
          border-color: rgba(255, 210, 100, 0.25);
          color: rgba(255, 210, 100, 0.8);
        }

        .btn-yawn:hover:not(:disabled),
        .btn-yawn.selected {
          background: rgba(255, 210, 100, 0.18);
          border-color: rgba(255, 210, 100, 0.7);
          color: #ffd264;
          box-shadow: 0 0 16px rgba(255, 210, 100, 0.15);
        }

        .btn-normal {
          background: rgba(136, 170, 255, 0.08);
          border-color: rgba(136, 170, 255, 0.25);
          color: rgba(136, 170, 255, 0.8);
        }

        .btn-normal:hover:not(:disabled),
        .btn-normal.selected {
          background: rgba(136, 170, 255, 0.18);
          border-color: rgba(136, 170, 255, 0.7);
          color: #88aaff;
          box-shadow: 0 0 16px rgba(136, 170, 255, 0.15);
        }

        .btn-grooming {
          background: rgba(220, 130, 255, 0.08);
          border-color: rgba(220, 130, 255, 0.25);
          color: rgba(220, 130, 255, 0.8);
        }

        .btn-grooming:hover:not(:disabled),
        .btn-grooming.selected {
          background: rgba(220, 130, 255, 0.18);
          border-color: rgba(220, 130, 255, 0.7);
          color: #dc82ff;
          box-shadow: 0 0 16px rgba(220, 130, 255, 0.15);
        }

        .btn-standing {
          background: rgba(255, 160, 60, 0.08);
          border-color: rgba(255, 160, 60, 0.25);
          color: rgba(255, 160, 60, 0.8);
        }

        .btn-standing:hover:not(:disabled),
        .btn-standing.selected {
          background: rgba(255, 160, 60, 0.18);
          border-color: rgba(255, 160, 60, 0.7);
          color: #ff9f3c;
          box-shadow: 0 0 16px rgba(255, 160, 60, 0.15);
        }

        .label-btn-icon { font-size: 20px; }
        .label-btn-text { flex: 1; text-align: left; }

        .label-btn-key {
          font-size: 10px;
          padding: 2px 6px;
          border-radius: 3px;
          background: rgba(255, 255, 255, 0.07);
          border: 1px solid rgba(255, 255, 255, 0.12);
          opacity: 0.6;
        }

        /* Clip action row */
        .clip-actions {
          display: flex;
          gap: 8px;
          align-items: center;
          justify-content: center;
        }

        .remove-label-btn {
          font-size: 11px;
          padding: 4px 12px;
          background: none;
          border: 1px solid var(--border);
          color: var(--text-muted);
          border-radius: var(--radius);
          font-family: var(--font-mono);
          cursor: pointer;
          transition: all 0.15s;
        }

        .remove-label-btn:hover { color: var(--red); border-color: var(--red-dim); }

        .delete-clip-btn {
          font-size: 11px;
          padding: 4px 12px;
          background: none;
          border: 1px solid var(--border);
          color: var(--text-muted);
          border-radius: var(--radius);
          font-family: var(--font-mono);
          cursor: pointer;
          transition: all 0.15s;
        }

        .delete-clip-btn:hover { color: var(--red); border-color: var(--red-dim); background: rgba(255,80,80,0.06); }

        /* Hint */
        .shortcut-hint {
          text-align: center;
          font-size: 10px;
          color: var(--text-muted);
          letter-spacing: 0.05em;
          opacity: 0.6;
        }

        .suggestion-hint {
          text-align: center;
          font-size: 12px;
          color: var(--text-secondary);
          background: var(--bg-card);
          border: 1px dashed var(--border);
          border-radius: var(--radius);
          padding: 6px 10px;
        }
        .suggestion-hint strong { text-transform: uppercase; }

        /* Filmstrip */
        .filmstrip {
          display: flex;
          flex-direction: column;
          gap: 4px;
          max-height: 680px;
          overflow-y: auto;
          overflow-x: visible;
          padding-right: 4px;
        }

        .filmstrip::-webkit-scrollbar { width: 4px; }
        .filmstrip::-webkit-scrollbar-track { background: transparent; }
        .filmstrip::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .strip-thumb {
          position: relative;
          border-radius: var(--radius);
          overflow: visible;
          border: 2px solid transparent;
          cursor: default;
          transition: border-color 0.15s;
          padding: 0;
          background: var(--bg-card);
          flex-shrink: 0;
        }

        .strip-thumb:hover        { border-color: var(--border-light); }
        .strip-thumb:hover .strip-menu-btn { opacity: 1; }

        /* Three-dot button */
        .strip-menu-btn {
          position: absolute;
          top: 3px;
          right: 3px;
          width: 20px;
          height: 20px;
          border-radius: 4px;
          background: rgba(0, 0, 0, 0.65);
          backdrop-filter: blur(4px);
          border: 1px solid rgba(255,255,255,0.1);
          color: #fff;
          font-size: 13px;
          line-height: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          opacity: 0;
          transition: opacity 0.15s, background 0.15s;
          z-index: 20;
          padding: 0;
        }

        .strip-menu-btn:hover { background: rgba(0,0,0,0.85); }

        /* Dropdown */
        .strip-menu {
          position: absolute;
          top: 26px;
          right: 0;
          background: var(--bg-surface);
          border: 1px solid var(--border-light);
          border-radius: var(--radius);
          box-shadow: 0 4px 16px rgba(0,0,0,0.4);
          z-index: 100;
          min-width: 110px;
          overflow: hidden;
        }

        .strip-menu-item {
          display: flex;
          align-items: center;
          gap: 6px;
          width: 100%;
          padding: 7px 12px;
          font-size: 11px;
          font-family: var(--font-mono);
          color: var(--text-secondary);
          background: none;
          border: none;
          text-align: left;
          cursor: pointer;
          transition: background 0.1s, color 0.1s;
        }

        .strip-menu-item:hover { background: var(--bg-card-hover); color: var(--text-primary); }
        .strip-menu-delete:hover { color: var(--red); }
        .strip-current            { border-color: var(--accent) !important; }
        .strip-zoomies:not(.strip-current)    { border-color: rgba(125, 255, 125, 0.35); }
        .strip-yawn:not(.strip-current)     { border-color: rgba(255, 210, 100, 0.35); }
        .strip-normal:not(.strip-current)   { border-color: rgba(136, 170, 255, 0.35); }
        .strip-grooming:not(.strip-current) { border-color: rgba(220, 130, 255, 0.35); }
        .strip-standing:not(.strip-current) { border-color: rgba(255, 160,  60, 0.35); }

        .strip-video {
          width: 100%;
          aspect-ratio: 16 / 9;
          object-fit: cover;
          display: block;
          pointer-events: auto;
          cursor: pointer;
          border-radius: calc(var(--radius) - 2px);
          overflow: hidden;
        }

        .strip-badge {
          position: absolute;
          bottom: 3px;
          right: 3px;
          font-size: 9px;
          font-weight: 700;
          padding: 1px 5px;
          border-radius: 3px;
          font-family: var(--font-mono);
        }

        .strip-badge-zoomies    { background: rgba(0,0,0,0.75); color: #7dff7d; }
        .strip-badge-yawn     { background: rgba(0,0,0,0.75); color: #ffd264; }
        .strip-badge-normal   { background: rgba(0,0,0,0.75); color: #88aaff; }
        .strip-badge-grooming { background: rgba(0,0,0,0.75); color: #dc82ff; }
        .strip-badge-standing { background: rgba(0,0,0,0.75); color: #ff9f3c; }

        /* Import panel */
        .import-panel {
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          overflow: hidden;
          background: var(--bg-card);
        }

        .import-toggle {
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 16px;
          background: none;
          border: none;
          color: var(--text-secondary);
          font-family: var(--font-mono);
          font-size: 12px;
          cursor: pointer;
          text-align: left;
          transition: color 0.15s;
        }

        .import-toggle:hover { color: var(--text-primary); }

        .import-toggle-sub {
          font-size: 10px;
          color: var(--text-muted);
          letter-spacing: 0.05em;
        }

        .import-body {
          padding: 12px 16px 16px;
          border-top: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .drop-zone {
          border: 2px dashed var(--border);
          border-radius: var(--radius-lg);
          padding: 28px 20px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
          cursor: pointer;
          transition: all 0.2s;
          text-align: center;
        }

        .drop-zone:hover,
        .drop-zone-active {
          border-color: var(--accent-soft);
          background: rgba(200, 169, 110, 0.04);
        }

        .drop-icon { font-size: 28px; }

        .drop-text {
          font-size: 13px;
          color: var(--text-secondary);
        }

        .drop-sub {
          font-size: 10px;
          color: var(--text-muted);
        }

        /* File list */
        .import-file-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
          max-height: 180px;
          overflow-y: auto;
        }

        .import-file-row {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 5px 8px;
          border-radius: var(--radius);
          background: var(--bg-surface);
          font-size: 11px;
          font-family: var(--font-mono);
        }

        .import-file-icon { width: 16px; text-align: center; flex-shrink: 0; }
        .import-file-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-secondary); }
        .import-file-size { color: var(--text-muted); flex-shrink: 0; }
        .import-file-err  { color: var(--red); font-size: 10px; flex-shrink: 0; }

        .status-ok         { opacity: 0.6; }
        .status-converting { opacity: 0.8; }
        .status-error      .import-file-name { color: var(--red); }

        .import-file-remove {
          background: none;
          border: none;
          color: var(--text-muted);
          font-size: 10px;
          cursor: pointer;
          padding: 0 2px;
          flex-shrink: 0;
          transition: color 0.15s;
        }

        .import-file-remove:hover { color: var(--red); }

        /* Import actions */
        .import-actions {
          display: flex;
          justify-content: flex-end;
          gap: 8px;
        }

        .import-clear-btn {
          font-size: 11px;
          padding: 5px 12px;
          border-radius: var(--radius);
          background: none;
          border: 1px solid var(--border);
          color: var(--text-muted);
          font-family: var(--font-mono);
          cursor: pointer;
          transition: all 0.15s;
        }

        .import-clear-btn:hover { color: var(--text-secondary); border-color: var(--border-light); }

        .import-upload-btn {
          font-size: 11px;
          padding: 5px 16px;
          border-radius: var(--radius);
          background: var(--accent);
          border: 1px solid var(--accent);
          color: var(--bg-deep);
          font-family: var(--font-mono);
          font-weight: 600;
          cursor: pointer;
          transition: all 0.15s;
        }

        .import-upload-btn:hover:not(:disabled) {
          background: color-mix(in srgb, var(--accent) 85%, white);
        }

        .import-upload-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Responsive */
        @media (max-width: 900px) {
          .studio-body {
            grid-template-columns: 1fr;
          }

          .filmstrip {
            flex-direction: row;
            max-height: none;
            overflow-x: auto;
            overflow-y: visible;
          }

          .strip-thumb { width: 120px; flex-shrink: 0; }
        }
      `}</style>
    </div>
  )
}
