import React, { useState, useEffect, useCallback, useRef } from "react";
import { ALL_CODES, LABEL_META, KEY_TO_LABEL } from "../labels";

// Ethogram v2 has eight codes where v1 had five, and the v1 studio hardcoded a
// button, a stat chip, a progress segment, a border class and a badge class
// per label. Everything below is generated from LABEL_META instead, so the next
// ethogram change is a one-line edit in labels.js rather than a dozen here.
const ACCEPTED_EXTENSIONS = ".mp4,.mov,.avi,.mkv,.webm,.m4v";

// ── Import Panel ────────────────────────────────────────────────────────────
function ImportPanel({ onImportDone }) {
  const [dragging, setDragging] = useState(false);
  const [files, setFiles] = useState([]); // [{ file, status, detail }]
  const [uploading, setUploading] = useState(false);
  const [open, setOpen] = useState(false);
  const inputRef = useRef(null);

  const addFiles = (incoming) => {
    const videoFiles = Array.from(incoming).filter(
      (f) =>
        f.type.startsWith("video/") ||
        /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name),
    );
    setFiles((prev) => {
      const existingNames = new Set(prev.map((e) => e.file.name));
      const newEntries = videoFiles
        .filter((f) => !existingNames.has(f.name))
        .map((f) => ({ file: f, status: "pending", detail: "" }));
      return [...prev, ...newEntries];
    });
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const removeFile = (name) =>
    setFiles((prev) => prev.filter((e) => e.file.name !== name));

  const upload = async () => {
    if (uploading) return;
    // Take the pending entries from the CURRENT state, not from what setFiles is
    // about to produce: the state update below has not committed yet, so reading
    // `files` back for status 'converting' would match nothing and upload nothing.
    const pending = files.filter((e) => e.status === "pending");
    if (pending.length === 0) return;
    setUploading(true);

    setFiles((prev) =>
      prev.map((e) =>
        e.status === "pending" ? { ...e, status: "converting" } : e,
      ),
    );

    const formData = new FormData();
    pending.forEach((e) => formData.append("videos", e.file));
    const sent = new Set(pending.map((e) => e.file.name));

    try {
      const res = await fetch("/api/import", {
        method: "POST",
        body: formData,
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok && !data.results) {
        throw new Error(data.error || `Import failed (HTTP ${res.status})`);
      }

      setFiles((prev) =>
        prev.map((e) => {
          if (!sent.has(e.file.name)) return e;
          const result = data.results?.find((r) => r.original === e.file.name);
          if (!result)
            return { ...e, status: "error", detail: "No result returned" };
          return { ...e, status: result.status, detail: result.detail || "" };
        }),
      );

      onImportDone();
    } catch (err) {
      setFiles((prev) =>
        prev.map((e) =>
          sent.has(e.file.name) && e.status === "converting"
            ? { ...e, status: "error", detail: err.message }
            : e,
        ),
      );
    } finally {
      setUploading(false);
    }
  };

  const clearDone = () =>
    setFiles((prev) => prev.filter((e) => e.status !== "ok"));

  const pendingCount = files.filter((e) => e.status === "pending").length;
  const doneCount = files.filter((e) => e.status === "ok").length;

  return (
    <div className="import-panel">
      <button className="import-toggle" onClick={() => setOpen((o) => !o)}>
        <span>{open ? "▾" : "▸"} Import Footage</span>
        <span className="import-toggle-sub">mp4, mov, avi, mkv, webm</span>
      </button>

      {open && (
        <div className="import-body">
          {/* Drop zone */}
          <div
            className={`drop-zone ${dragging ? "drop-zone-active" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <span className="drop-icon">📂</span>
            <p className="drop-text">
              Drop video files here or click to browse
            </p>
            <p className="drop-sub">
              Up to 20 files · 500 MB each · ffmpeg converts automatically
            </p>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={ACCEPTED_EXTENSIONS}
              style={{ display: "none" }}
              onChange={(e) => addFiles(e.target.files)}
            />
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="import-file-list">
              {files.map(({ file, status, detail }) => (
                <div
                  key={file.name}
                  className={`import-file-row status-${status}`}
                >
                  <span className="import-file-icon">
                    {status === "pending" && "⏳"}
                    {status === "converting" && "⚙️"}
                    {status === "ok" && "✓"}
                    {status === "error" && "✕"}
                  </span>
                  <span className="import-file-name" title={file.name}>
                    {file.name}
                  </span>
                  <span className="import-file-size">
                    {formatSize(file.size)}
                  </span>
                  {status === "error" && (
                    <span className="import-file-err" title={detail}>
                      failed
                    </span>
                  )}
                  {status === "pending" && (
                    <button
                      className="import-file-remove"
                      onClick={() => removeFile(file.name)}
                    >
                      ✕
                    </button>
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
                  ? "⚙️ Converting…"
                  : `↑ Import ${pendingCount} clip${pendingCount !== 1 ? "s" : ""}`}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatDate(isoStr) {
  return new Date(isoStr).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Lazy filmstrip thumbnail ──────────────────────────────────────────
// Mounting a <video> for every clip is what froze the studio: with a few
// hundred recordings the browser opened that many media pipelines at once,
// each demuxing a ~1 MB webm on the main thread. A thumb now only carries a
// src while it is on (or near) screen, so at most a screenful decodes at a time.

// Retention policy for a filmstrip thumbnail: hold a decoded <video> only
// while the thumb is on (or near) screen, and drop it otherwise. Strict, so
// memory stays flat no matter how many clips accumulate.
//
// Anti-thrash is handled by the observer's rootMargin rather than by an
// intersectionRatio band. A ratio band would be dead weight here: when a
// target is not intersecting its ratio is exactly 0, and the observer's
// default threshold of [0] only fires on the crossing into and out of
// intersection — it never reports intermediate ratios to test against.
function shouldHoldVideo(entry) {
  return Boolean(entry && entry.isIntersecting);
}

function StripThumb({ filename, onClick }) {
  const holderRef = useRef(null);
  const [holding, setHolding] = useState(false);

  useEffect(() => {
    const el = holderRef.current;
    if (!el) return;
    // rootMargin pre-loads a little above/below the fold so scrolling the
    // strip doesn't show a wall of empty placeholders.
    const io = new IntersectionObserver(
      ([entry]) => setHolding(shouldHoldVideo(entry)),
      { rootMargin: "300px 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={holderRef} className="strip-video-holder" onClick={onClick}>
      {holding ? (
        <video
          muted
          preload="metadata"
          src={`/recordings/${filename}#t=0.5`}
          className="strip-video"
        />
      ) : (
        <div className="strip-video strip-video-placeholder" />
      )}
    </div>
  );
}

export default function LabelingStudio() {
  const [recordings, setRecordings] = useState([]); // [{ filename, createdAt, size }]
  const [labels, setLabels] = useState({}); // { filename: <one of ALL_CODES> }
  const [suggestions, setSuggestions] = useState({}); // agent-predicted labels awaiting review (not training data)
  const [index, setIndex] = useState(0);
  const [filter, setFilter] = useState("all"); // 'all' | 'unlabeled' | <one of ALL_CODES>
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [openMenu, setOpenMenu] = useState(null); // filename of open three-dot menu
  const videoRef = useRef(null);

  // ── Load recordings + labels ──────────────────────────────────────────────
  const loadData = useCallback(async () => {
    try {
      const [recRes, labRes] = await Promise.all([
        fetch("/api/recordings"),
        fetch("/api/labels"),
      ]);
      const recData = await recRes.json();
      const labData = await labRes.json();
      setRecordings(recData.recordings || []);
      setLabels(labData.labels || {});
      setSuggestions(labData.suggestions || {});
    } catch (err) {
      console.error("Failed to load labeling data:", err);
    }
  }, []);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [recRes, labRes] = await Promise.all([
          fetch("/api/recordings"),
          fetch("/api/labels"),
        ]);
        const recData = await recRes.json();
        const labData = await labRes.json();
        setRecordings(recData.recordings || []);
        setLabels(labData.labels || {});
        setSuggestions(labData.suggestions || {});
      } catch (err) {
        console.error("Failed to load labeling data:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // ── Filtered + sorted clip list ───────────────────────────────────────────
  const filtered = recordings.filter((r) => {
    if (filter === "unlabeled") return !labels[r.filename];
    if (filter !== "all") return labels[r.filename] === filter;
    return true;
  });

  const current = filtered[index] || null;

  // Reset index when filter changes
  useEffect(() => {
    setIndex(0);
  }, [filter]);

  // Replay video when clip changes
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load();
      videoRef.current.play().catch(() => {});
    }
  }, [current?.filename]);

  // ── Label actions ─────────────────────────────────────────────────────────
  const applyLabel = useCallback(
    async (label) => {
      if (!current || saving) return;
      setSaving(true);
      try {
        const res = await fetch(`/api/labels/${current.filename}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label }),
        });
        if (!res.ok) {
          const detail =
            res.status === 401
              ? "Not logged in. Your session expired. Reload and log in again."
              : `Save failed (HTTP ${res.status})`;
          alert(detail);
          return; // do NOT optimistically update — the label did not save
        }
        setLabels((prev) => ({ ...prev, [current.filename]: label }));
        // Labeling resolves the agent's suggestion (server clears it too)
        setSuggestions((prev) => {
          const next = { ...prev };
          delete next[current.filename];
          return next;
        });
        // Advance to next clip automatically
        setIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      } catch (err) {
        console.error("Label save failed:", err);
        alert("Label save failed. Check your connection.");
      } finally {
        setSaving(false);
      }
    },
    [current, saving, filtered.length],
  );

  const removeLabel = useCallback(async () => {
    if (!current || !labels[current.filename]) return;
    try {
      await fetch(`/api/labels/${current.filename}`, { method: "DELETE" });
      setLabels((prev) => {
        const next = { ...prev };
        delete next[current.filename];
        return next;
      });
    } catch (err) {
      console.error("Label remove failed:", err);
    }
  }, [current, labels]);

  const deleteClip = useCallback(
    async (filename) => {
      const target = filename || current?.filename;
      if (!target) return;
      if (
        !window.confirm(
          `Delete "${target}" permanently? This cannot be undone.`,
        )
      )
        return;
      setOpenMenu(null);
      try {
        await fetch(`/api/recordings/${target}`, { method: "DELETE" });
        await fetch(`/api/labels/${target}`, { method: "DELETE" }).catch(
          () => {},
        );
        setLabels((prev) => {
          const next = { ...prev };
          delete next[target];
          return next;
        });
        setRecordings((prev) => prev.filter((r) => r.filename !== target));
        setIndex((prev) => Math.max(0, prev > 0 ? prev - 1 : 0));
      } catch (err) {
        console.error("Delete failed:", err);
      }
    },
    [current],
  );

  const downloadClip = useCallback(
    (filename) => {
      const target = filename || current?.filename;
      if (!target) return;
      setOpenMenu(null);
      const a = document.createElement("a");
      a.href = `/recordings/${target}`;
      a.download = target;
      a.click();
    },
    [current],
  );

  // ── Close three-dot menu on outside click ────────────────────────────────
  useEffect(() => {
    if (!openMenu) return;
    const close = () => setOpenMenu(null);
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, [openMenu]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "BUTTON") return;
      const byKey = KEY_TO_LABEL[e.key];
      if (byKey) applyLabel(byKey);
      if (e.key === "ArrowRight" || e.key === "d")
        setIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      if (e.key === "ArrowLeft" || e.key === "a")
        setIndex((prev) => Math.max(prev - 1, 0));
      if (e.key === "Delete" || e.key === "Backspace") removeLabel();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [applyLabel, removeLabel, filtered.length]);

  // ── Export labels.json ────────────────────────────────────────────────────
  const exportLabels = () => {
    const blob = new Blob([JSON.stringify(labels, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "labels.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Stats ─────────────────────────────────────────────────────────────────
  const totalCount = recordings.length;
  // One pass over the label map, not one pass per code.
  const counts = ALL_CODES.reduce((acc, id) => ({ ...acc, [id]: 0 }), {});
  for (const l of Object.values(labels)) {
    if (l in counts) counts[l] += 1;
  }
  const labeledCount = ALL_CODES.reduce((n, id) => n + counts[id], 0);
  const pctDone =
    totalCount > 0 ? Math.round((labeledCount / totalCount) * 100) : 0;

  if (loading) {
    return (
      <div className="studio-loading">
        <span>🐇</span>
        <p>Loading clips…</p>
        <style>{`.studio-loading { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:12px; min-height:400px; color:var(--text-muted); font-size:14px; }`}</style>
      </div>
    );
  }

  if (recordings.length === 0) {
    return (
      <div className="studio-empty">
        <span>🎥</span>
        <p>No recordings to label yet.</p>
        <p className="sub">
          Record clips from the Camera tab, or import footage below.
        </p>
        <ImportPanel onImportDone={loadData} />
        <style>{`
          .studio-empty { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; min-height:400px; color:var(--text-muted); }
          .studio-empty span { font-size:48px; }
          .studio-empty p { font-size:14px; }
          .studio-empty .sub { font-size:12px; }
        `}</style>
      </div>
    );
  }

  const currentLabel = current ? labels[current.filename] : null;
  const currentSuggestion =
    current && !currentLabel ? suggestions[current.filename] : null;

  return (
    <div className="studio">
      <ImportPanel onImportDone={loadData} />

      {/* ── Progress header ─────────────────────────────────────────────── */}
      <div className="studio-header">
        <div className="studio-title">Label Studio</div>
        <div className="studio-stats">
          {ALL_CODES.map((id) => (
            <span
              key={id}
              className="stat-chip"
              style={{
                background: `${LABEL_META[id].color}1a`,
                borderColor: `${LABEL_META[id].color}4d`,
                color: LABEL_META[id].color,
              }}
            >
              {counts[id]} {LABEL_META[id].short}
            </span>
          ))}
          <span className="stat-chip stat-unlabeled">
            {totalCount - labeledCount} unlabeled
          </span>
        </div>
        <button
          className="export-btn"
          onClick={exportLabels}
          disabled={labeledCount === 0}
        >
          ↓ Export labels.json
        </button>
      </div>

      {/* ── Progress bar ────────────────────────────────────────────────── */}
      <div
        className="progress-track"
        title={`${labeledCount} of ${totalCount} labeled`}
      >
        {ALL_CODES.map((id) => (
          <div
            key={id}
            className="progress-seg"
            title={`${counts[id]} ${LABEL_META[id].name}`}
            style={{
              background: LABEL_META[id].color,
              width: `${totalCount > 0 ? (counts[id] / totalCount) * 100 : 0}%`,
            }}
          />
        ))}
      </div>
      <div className="progress-label">
        {pctDone}% labeled, {labeledCount} / {totalCount} clips
      </div>

      {/* ── Filter tabs ─────────────────────────────────────────────────── */}
      <div className="filter-tabs">
        {["all", "unlabeled", ...ALL_CODES].map((f) => (
          <button
            key={f}
            className={`filter-tab ${filter === f ? "active" : ""}`}
            onClick={() => setFilter(f)}
          >
            {f === "all"
              ? `All (${totalCount})`
              : f === "unlabeled"
                ? `Unlabeled (${totalCount - labeledCount})`
                : `${LABEL_META[f].short} (${counts[f]})`}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="no-clips">No clips match this filter.</div>
      ) : (
        <div className="studio-body">
          {/* ── Video player ──────────────────────────────────────────────── */}
          <div className="player-section">
            <div
              className="player-wrap"
              style={
                currentLabel && LABEL_META[currentLabel]
                  ? { borderColor: `${LABEL_META[currentLabel].color}99` }
                  : undefined
              }
            >
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
                <div
                  className="current-label-badge"
                  style={{
                    color: LABEL_META[currentLabel]?.color,
                    borderColor: `${LABEL_META[currentLabel]?.color}66`,
                  }}
                >
                  {LABEL_META[currentLabel]?.icon}{" "}
                  {LABEL_META[currentLabel]?.name.toUpperCase()}
                </div>
              )}
            </div>

            {/* Clip metadata */}
            {current && (
              <div className="clip-meta">
                <span className="clip-name">{current.filename}</span>
                <span className="clip-detail">
                  {formatDate(current.createdAt)} · {formatSize(current.size)}
                </span>
              </div>
            )}

            {/* Agent suggestion — a prediction awaiting review, not a saved label */}
            {currentSuggestion && (
              <div className="suggestion-hint">
                🤖 Agent suggests <strong>{currentSuggestion}</strong>. Label to
                confirm or correct.
              </div>
            )}

            {/* Navigation */}
            <div className="nav-row">
              <button
                className="nav-btn"
                onClick={() => setIndex((prev) => Math.max(prev - 1, 0))}
                disabled={index === 0}
              >
                ← Prev
              </button>
              <span className="nav-counter">
                {index + 1} / {filtered.length}
              </span>
              <button
                className="nav-btn"
                onClick={() =>
                  setIndex((prev) => Math.min(prev + 1, filtered.length - 1))
                }
                disabled={index === filtered.length - 1}
              >
                Next →
              </button>
            </div>

            {/* Label buttons */}
            <div className="label-buttons">
              {ALL_CODES.map((id) => (
                <button
                  key={id}
                  className={`label-btn ${currentLabel === id ? "selected" : ""}`}
                  style={{
                    borderColor:
                      currentLabel === id
                        ? LABEL_META[id].color
                        : `${LABEL_META[id].color}40`,
                    color: LABEL_META[id].color,
                    background:
                      currentLabel === id ? `${LABEL_META[id].color}26` : undefined,
                  }}
                  onClick={() => applyLabel(id)}
                  disabled={saving}
                >
                  <span className="label-btn-icon">{LABEL_META[id].icon}</span>
                  <span className="label-btn-text">{LABEL_META[id].name}</span>
                  <span className="label-btn-key">{LABEL_META[id].key}</span>
                </button>
              ))}
            </div>

            <div className="clip-actions">
              {currentLabel && (
                <button className="remove-label-btn" onClick={removeLabel}>
                  ✕ Remove label
                </button>
              )}
              <button
                className="delete-clip-btn"
                onClick={() => deleteClip()}
                title="Permanently delete this clip"
              >
                🗑 Delete clip
              </button>
            </div>

            <div className="shortcut-hint">
              Arrow keys to navigate · 1-7 to label, 0 for out of view · Delete
              to clear
            </div>
          </div>

          {/* ── Filmstrip ─────────────────────────────────────────────────── */}
          <div className="filmstrip">
            {filtered.map((r, i) => {
              const lbl = labels[r.filename];
              const menuOpen = openMenu === r.filename;
              return (
                <div
                  key={r.filename}
                  className={`strip-thumb ${i === index ? "strip-current" : ""} ${lbl ? "" : "strip-unlabeled"}`}
                  style={
                    lbl && LABEL_META[lbl]
                      ? { borderColor: `${LABEL_META[lbl].color}99` }
                      : undefined
                  }
                  title={`${r.filename} — ${lbl ? LABEL_META[lbl].name : "unlabeled"}`}
                >
                  {/* Thumbnail — clicking selects the clip */}
                  <StripThumb
                    filename={r.filename}
                    onClick={() => setIndex(i)}
                  />

                  {/* Label badge */}
                  {lbl && (
                    <span
                      className="strip-badge"
                      style={{
                        color: LABEL_META[lbl]?.color,
                        borderColor: `${LABEL_META[lbl]?.color}66`,
                      }}
                      title={LABEL_META[lbl]?.name}
                    >
                      {LABEL_META[lbl]?.icon}
                    </span>
                  )}

                  {/* Three-dot menu button */}
                  <button
                    className="strip-menu-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      setOpenMenu(menuOpen ? null : r.filename);
                    }}
                    title="Options"
                  >
                    ⋯
                  </button>

                  {/* Dropdown */}
                  {menuOpen && (
                    <div
                      className="strip-menu"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        className="strip-menu-item"
                        onClick={() => downloadClip(r.filename)}
                      >
                        ↓ Download
                      </button>
                      <button
                        className="strip-menu-item strip-menu-delete"
                        onClick={() => deleteClip(r.filename)}
                      >
                        🗑 Delete
                      </button>
                    </div>
                  )}
                </div>
              );
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

        /* Per-label colour is applied inline from LABEL_META (see labels.js).
           v1 had one CSS class per label here; eight codes made that a
           maintenance trap, and adding a ninth would mean editing five rules. */
        .stat-chip {
          font-size: 11px;
          padding: 3px 10px;
          border-radius: 20px;
          border: 1px solid transparent;
        }

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


        .progress-seg { height: 100%; transition: width 0.3s ease; }

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


        .player-video {
          width: 100%;
          height: 100%;
          object-fit: contain;
          display: block;
        }

        .current-label-badge {
          background: rgba(0, 0, 0, 0.7);
          border: 1px solid transparent;
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
        .label-btn { background: rgba(255, 255, 255, 0.03); }
        .label-btn:hover:not(:disabled) { filter: brightness(1.25); }

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

        .strip-video-holder {
          display: block;
          width: 100%;
        }

        .strip-video-placeholder {
          background: var(--bg-card-hover);
          border: 1px solid var(--border);
        }

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
          background: rgba(0, 0, 0, 0.75);
          border: 1px solid transparent;
          position: absolute;
          bottom: 3px;
          right: 3px;
          font-size: 9px;
          font-weight: 700;
          padding: 1px 5px;
          border-radius: 3px;
          font-family: var(--font-mono);
        }


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
  );
}
