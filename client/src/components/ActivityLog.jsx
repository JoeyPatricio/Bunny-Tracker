import React, { useEffect, useRef, useState } from 'react'
import { LABEL_COLOR, LABEL_PHRASE } from '../labels.js'

/**
 * ActivityLog
 * Scrollable list of the agent's recent behavior predictions, polled from
 * /api/predictions (the same feed AgentFeed.jsx uses for its live badge).
 * Only behavior-label changes appear here — the agent's feed doesn't carry
 * raw motion-level events, so this no longer shows generic motion blips.
 */
export default function ActivityLog() {
  const [events, setEvents] = useState([])
  const bottomRef = useRef(null)

  useEffect(() => {
    const poll = () =>
      fetch('/api/predictions')
        .then(r => r.json())
        .then(d => {
          const items = (d.events || []).map((ev, i) => ({
            id: `${ev.time}-${i}`,
            timestamp: new Date(ev.time),
            message: LABEL_PHRASE[ev.label] ?? ev.label,
            confidence: ev.confidence,
            color: LABEL_COLOR[ev.label],
          }))
          setEvents(items)
        })
        .catch(() => {})
    poll()
    const id = setInterval(poll, 3000)
    return () => clearInterval(id)
  }, [])

  // Auto-scroll to the latest entry, but only when a genuinely new one arrives.
  // The 3s poll rebuilds `events` as a fresh array every tick, so depending on
  // it directly re-scrolled every 3 seconds whether anything had changed or
  // not, yanking the view back down while you were reading older entries.
  const newestId = events.length ? events[events.length - 1].id : null
  useEffect(() => {
    if (newestId === null) return
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [newestId])

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  return (
    <div className="activity-log">
      <div className="log-header">
        <span className="log-title">Activity Log</span>
      </div>

      <div className="log-body">
        {events.length === 0 ? (
          <div className="log-empty">
            <span className="empty-icon">🌿</span>
            <span>No motion detected yet</span>
          </div>
        ) : (
          <>
            {events.map((event) => (
              <div key={event.id} className="log-entry log-entry-prediction">
                <span className="entry-time">{formatTime(event.timestamp)}</span>
                <span
                  className={`entry-dot${event.color ? ' colored' : ''}`}
                  style={event.color ? { background: event.color } : {}}
                />
                <span className="entry-msg" style={event.color ? { color: event.color } : {}}>
                  {event.message}
                </span>
                {event.confidence !== undefined && (
                  <span className="entry-level">{event.confidence}%</span>
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      <style>{`
        .activity-log {
          display: flex;
          flex-direction: column;
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          overflow: hidden;
          height: 100%;
          min-height: 200px;
          max-height: 340px;
        }

        .log-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 14px;
          border-bottom: 1px solid var(--border);
          background: var(--bg-surface);
        }

        .log-title {
          font-family: var(--font-display);
          font-size: 13px;
          color: var(--text-secondary);
          letter-spacing: 0.04em;
        }

        .log-body {
          flex: 1;
          overflow-y: auto;
          padding: 8px 0;
        }

        .log-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 8px;
          height: 100%;
          color: var(--text-muted);
          font-size: 12px;
          padding: 32px;
        }

        .empty-icon { font-size: 24px; }

        .log-entry {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 5px 14px;
          border-bottom: 1px solid color-mix(in srgb, var(--border) 50%, transparent);
          transition: background 0.1s;
        }

        .log-entry:hover {
          background: var(--bg-card-hover);
        }

        .log-entry:last-of-type {
          border-bottom: none;
        }

        .entry-time {
          color: var(--text-muted);
          font-size: 10px;
          white-space: nowrap;
          min-width: 72px;
        }

        .entry-dot {
          width: 4px;
          height: 4px;
          border-radius: 50%;
          background: var(--accent);
          flex-shrink: 0;
        }

        .entry-dot.colored {
          width: 6px;
          height: 6px;
        }

        .entry-msg {
          flex: 1;
          color: var(--text-secondary);
          font-size: 11px;
        }

        .entry-level {
          color: var(--text-muted);
          font-size: 10px;
          white-space: nowrap;
        }
      `}</style>
    </div>
  )
}
