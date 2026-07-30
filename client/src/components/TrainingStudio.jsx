import React, { useEffect, useState } from 'react'
import { LABEL_COLOR } from '../labels.js'

/**
 * TrainingStudio
 * Read-only panel: training moved from an in-browser TensorFlow.js pipeline
 * to a PyTorch CLI script (server_py/training/train.py) once the Python port
 * retrained the classifier on a rebuilt backbone. This just displays the
 * currently deployed model's metadata.
 */
export default function TrainingStudio() {
  const [info, setInfo] = useState(null) // { exists, savedAt, labels, valAcc, window, backups }
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/model')
      .then(r => r.json())
      .then(setInfo)
      .catch(() => setError('Could not load model info'))
  }, [])

  const labels = info?.labels ?? []

  return (
    <div className="training-studio">
      <div className="ts-header">
        <span className="ts-title">🧠 Model</span>
        <span className="ts-subtitle">Training is a CLI script now, not a browser tab</span>
      </div>

      <div className="ts-note">
        Retraining runs offline: <code>server_py/training/train.py</code>, then{' '}
        <code>export_onnx.py</code>. They read the labeled clips, retrain the classifier,
        and write the ONNX artifacts the camera agent loads — there's nothing to do here
        except see what's currently deployed.
      </div>

      {error && <div className="ts-error">{error}</div>}

      {info && (
        <div className="ts-info-row">
          {labels.map(l => (
            <div key={l} className="ts-label-chip" style={{ borderColor: LABEL_COLOR[l], color: LABEL_COLOR[l] }}>
              {l}
            </div>
          ))}
        </div>
      )}

      {info?.exists ? (
        <div className="ts-metrics">
          <div className="ts-metrics-header">Deployed model</div>
          <div className="ts-metric-card">
            <div className="ts-metric-val">{new Date(info.savedAt).toLocaleString()}</div>
            <div className="ts-metric-label">Exported at</div>
          </div>
          {typeof info.valAcc === 'number' && (
            <div className="ts-metric-card">
              <div className="ts-metric-val">{(info.valAcc * 100).toFixed(1)}%</div>
              <div className="ts-metric-label">Held-out accuracy</div>
            </div>
          )}
          {typeof info.window === 'number' && (
            <div className="ts-metric-card">
              <div className="ts-metric-val">{info.window}</div>
              <div className="ts-metric-label">Frames per window</div>
            </div>
          )}
        </div>
      ) : info && !error ? (
        <div className="ts-note">No model deployed yet.</div>
      ) : null}

      <style>{`
        .training-studio {
          display: flex;
          flex-direction: column;
          gap: 14px;
          padding: 16px;
          max-width: 900px;
          margin: 0 auto;
        }

        .ts-header {
          display: flex;
          align-items: baseline;
          gap: 10px;
        }
        .ts-title {
          font-family: var(--font-display);
          font-size: 18px;
          color: var(--text-primary);
        }
        .ts-subtitle {
          font-size: 11px;
          color: var(--text-muted);
        }

        .ts-note {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 12px 14px;
          font-size: 12px;
          color: var(--text-secondary);
          line-height: 1.5;
        }
        .ts-note code {
          font-family: monospace;
          background: var(--bg-surface);
          padding: 1px 5px;
          border-radius: 3px;
        }

        .ts-error {
          color: var(--red, #ff6b6b);
          font-size: 12px;
        }

        .ts-info-row {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .ts-label-chip {
          font-size: 11px;
          padding: 2px 8px;
          border: 1px solid;
          border-radius: 99px;
          font-family: var(--font-display);
        }

        .ts-metrics {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 14px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .ts-metrics-header {
          font-family: var(--font-display);
          font-size: 13px;
          color: var(--text-secondary);
        }
        .ts-metric-card {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 10px 16px;
          min-width: 100px;
        }
        .ts-metric-val {
          font-size: 18px;
          font-family: var(--font-display);
          color: var(--accent);
        }
        .ts-metric-label { font-size: 10px; color: var(--text-muted); margin-top: 2px; }
      `}</style>
    </div>
  )
}
