import React, { useState, useRef, useCallback } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as mobilenetModule from '@tensorflow-models/mobilenet'
import { LABELS, LABEL_COLOR } from '../labels.js'

// ── Constants ────────────────────────────────────────────────────────────────
const FRAMES_PER_CLIP = 8   // frames sampled per clip
// Augmented variants per TRAINING clip, on top of the original: a mirrored
// copy plus a brightness/contrast-jittered copy. Multiplies effective training
// data and lighting robustness at zero labeling cost. Validation clips are
// never augmented — the metric stays grounded in real footage.
const AUG_COPIES      = 2
const EMBEDDING_DIM   = 1280 // MobileNet v2 alpha=1.0 embedding size
// Classifier input = mean ‖ std of the frame embeddings. The mean captures
// pose/appearance; the std across frames captures how much the scene CHANGES
// over the clip — the motion signal that separates zoomies from resting,
// which a mean alone (let alone a single frame) throws away.
const FEATURE_DIM     = EMBEDDING_DIM * 2
// Upper bound only — training stops early (and keeps the best epoch's
// weights) once val_loss hasn't improved for PATIENCE epochs.
const EPOCHS          = 100
const PATIENCE        = 10
const BATCH_SIZE      = 16
const LEARNING_RATE   = 0.001

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Resolve a usable duration. Clips saved by the browser Record button
 * (MediaRecorder) report duration === Infinity until the video is seeked to the
 * end, which would make every seek time non-finite and throw. Seeking past the
 * end forces the real duration to materialize.
 */
function resolveDuration(video) {
  if (Number.isFinite(video.duration) && video.duration > 0) {
    return Promise.resolve(video.duration)
  }
  return new Promise(resolve => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked)
      const d = video.duration
      video.currentTime = 0
      resolve(Number.isFinite(d) && d > 0 ? d : 0)
    }
    video.addEventListener('seeked', onSeeked)
    video.currentTime = 1e101 // clamped to the end, triggers 'seeked'
  })
}

/** Extract N evenly-spaced frames from a video element as ImageData */
async function extractFrames(videoEl, n) {
  const canvas  = document.createElement('canvas')
  canvas.width  = 224
  canvas.height = 224
  const ctx     = canvas.getContext('2d')
  const frames  = []
  const duration = await resolveDuration(videoEl)

  for (let i = 0; i < n; i++) {
    const t = duration > 0 ? (i / (n - 1 || 1)) * duration * 0.95 : 0
    await seekTo(videoEl, t)
    ctx.drawImage(videoEl, 0, 0, 224, 224)
    frames.push(ctx.getImageData(0, 0, 224, 224))
  }
  return frames
}

function seekTo(video, time) {
  return new Promise(resolve => {
    const onSeeked = () => { video.removeEventListener('seeked', onSeeked); resolve() }
    video.addEventListener('seeked', onSeeked)
    video.currentTime = time
  })
}

/** Load a video from a src URL and return the HTMLVideoElement (metadata loaded) */
function loadVideo(src) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video')
    v.crossOrigin = 'anonymous'
    v.muted = true
    v.preload = 'auto'
    v.onloadeddata = () => resolve(v)
    v.onerror = reject
    v.src = src
    v.load()
  })
}

/**
 * Load MobileNet from our own server — the dashboard's CSP (connect-src
 * 'self') blocks tfhub.dev, and a local copy also works offline. Falls back
 * to the CDN if the local copy is missing (run
 * server/scripts/download-mobilenet.mjs to create it).
 * inputRange [0,1] is REQUIRED with modelUrl: it's what this TFHub model
 * expects, and the package only applies it automatically for CDN loads.
 */
async function loadMobilenet() {
  try {
    return await mobilenetModule.load({
      version: 2, alpha: 1.0,
      modelUrl: '/model/mobilenet/model.json',
      inputRange: [0, 1],
    })
  } catch {
    return mobilenetModule.load({ version: 2, alpha: 1.0 })
  }
}

/** Run mobilenet.infer on an ImageData, return 1-D tensor (1280 for MobileNet v2 alpha=1.0) */
function embedFrame(mobilenet, imageData) {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(imageData)
    return mobilenet.infer(img, true).squeeze() // [1280]
  })
}

/**
 * Source-group key for a clip. Clips cut from the same source video (or the
 * same recording session) share scene, lighting, and rabbit — if they straddle
 * the train/val split, validation accuracy measures "have I seen this video
 * before" instead of generalization.
 */
function groupKey(filename) {
  // Timestamped agent/manual clips: one group per calendar hour ≈ one session
  let m = filename.match(/(\d{4}-\d{2}-\d{2}T\d{2})/)
  if (m) return m[1]
  // Curated imports: recording-<tag>-srcNN-MM.webm → source video NN
  m = filename.match(/^recording-([a-z]+)-src(\d+)/i)
  if (m) return `${m[1]}-src${m[2]}`
  // Compilations / named sources: recording-<tag>-<name>-NN.webm
  m = filename.match(/^recording-([a-z]+)-([a-z]+)/i)
  if (m) return `${m[1]}-${m[2]}`
  return filename
}

/**
 * Group-aware stratified split at the CLIP level, done before feature
 * extraction. All clips from one source video / session stay on the same
 * side (no leakage), each class aims for ~valFraction of its clips in
 * validation, and augmentation can then be applied to training clips only.
 */
function splitByGroup(clips, valFraction = 0.2) {
  const targets = clips.map(([, label]) => LABELS.indexOf(label))
  const classTotal = Array(LABELS.length).fill(0)
  targets.forEach(t => { classTotal[t]++ })
  const quota = classTotal.map(c => Math.max(1, Math.round(c * valFraction)))

  const groupMap = new Map() // group key → clip indices
  clips.forEach(([filename], i) => {
    const g = groupKey(filename)
    if (!groupMap.has(g)) groupMap.set(g, [])
    groupMap.get(g).push(i)
  })
  const groupList = Array.from(groupMap.keys())
  tf.util.shuffle(groupList)

  const valClass = Array(LABELS.length).fill(0)
  let train = []
  let val   = []
  for (const g of groupList) {
    const idxs = groupMap.get(g)
    const counts = Array(LABELS.length).fill(0)
    idxs.forEach(i => { counts[targets[i]]++ })
    // Send the group to validation if some class there still needs val
    // samples and no class would overshoot its quota by more than 1.
    const needed = counts.some((c, k) => c > 0 && valClass[k] < quota[k])
    const fits   = counts.every((c, k) => c === 0 || valClass[k] + c <= quota[k] + 1)
    if (needed && fits) {
      val = val.concat(idxs)
      counts.forEach((c, k) => { valClass[k] += c })
    } else {
      train = train.concat(idxs)
    }
  }

  // Degenerate fallback (too few groups to stratify): plain random 80/20
  let fallback = false
  if (val.length === 0 || train.length === 0) {
    fallback = true
    const idx = Array.from(tf.util.createShuffledIndices(clips.length))
    const at = Math.floor(clips.length * (1 - valFraction))
    train = idx.slice(0, at)
    val   = idx.slice(at)
    valClass.fill(0)
    val.forEach(i => { valClass[targets[i]]++ })
  }

  return {
    train: train.map(i => clips[i]),
    val:   val.map(i => clips[i]),
    groups: groupMap.size,
    valPerClass: valClass,
    totalPerClass: classTotal,
    fallback,
  }
}

// ── Frame augmentation (training clips only) ─────────────────────────────────
const clamp8 = v => (v < 0 ? 0 : v > 255 ? 255 : v)

function flipImageData(img) {
  const { width: w, height: h, data } = img
  const out = new ImageData(w, h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const src = (y * w + x) * 4
      const dst = (y * w + (w - 1 - x)) * 4
      out.data[dst]     = data[src]
      out.data[dst + 1] = data[src + 1]
      out.data[dst + 2] = data[src + 2]
      out.data[dst + 3] = data[src + 3]
    }
  }
  return out
}

function jitterImageData(img, brightness, contrast) {
  const out = new ImageData(img.width, img.height)
  const d = img.data
  const o = out.data
  for (let i = 0; i < d.length; i += 4) {
    o[i]     = clamp8((d[i]     - 128) * contrast + 128 + brightness)
    o[i + 1] = clamp8((d[i + 1] - 128) * contrast + 128 + brightness)
    o[i + 2] = clamp8((d[i + 2] - 128) * contrast + 128 + brightness)
    o[i + 3] = 255
  }
  return out
}

/**
 * Original + AUG_COPIES augmented variants of a clip's frames. Each variant
 * applies ONE transform consistently across all frames (whole-clip mirror,
 * whole-clip lighting shift) — like a different camera placement or time of
 * day, without disturbing the frame-to-frame motion signal the std features
 * depend on.
 */
function makeVariants(frames) {
  const variants = [frames, frames.map(flipImageData)]
  while (variants.length < AUG_COPIES + 1) {
    const brightness = Math.random() * 50 - 25   // ±25 levels
    const contrast   = 0.85 + Math.random() * 0.35 // 0.85–1.2×
    const alsoFlip   = Math.random() < 0.5
    variants.push(frames.map(f => {
      const j = jitterImageData(f, brightness, contrast)
      return alsoFlip ? flipImageData(j) : j
    }))
  }
  return variants
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function TrainingStudio() {
  const [phase, setPhase]           = useState('idle')   // idle | loading-model | extracting | training | done | error
  const [log, setLog]               = useState([])
  const [progress, setProgress]     = useState({ current: 0, total: 0, label: '' })
  const [metrics, setMetrics]       = useState(null)     // { accuracy, valAccuracy, confMatrix }
  const [modelInfo, setModelInfo]   = useState(null)     // saved model metadata
  const stopRef                     = useRef(false)

  const addLog = useCallback((msg, type = 'info') => {
    setLog(prev => [...prev, { msg, type, time: new Date().toLocaleTimeString() }])
  }, [])

  // ── Main training pipeline ─────────────────────────────────────────────────
  const runTraining = useCallback(async () => {
    stopRef.current = false
    setPhase('loading-model')
    setLog([])
    setMetrics(null)

    try {
      // 1 ── Fetch labeled clips from server
      addLog('Fetching labels from server…')
      const labelsRes = await fetch('/api/labels')
      if (labelsRes.status === 401) throw new Error('Session expired. Reload and log in again.')
      if (!labelsRes.ok) throw new Error(`Could not load labels (HTTP ${labelsRes.status})`)
      const { labels } = await labelsRes.json()
      const entries = Object.entries(labels).filter(([, l]) => LABELS.includes(l))
      addLog(`Found ${entries.length} labeled clips`)

      const recRes = await fetch('/api/recordings')
      if (!recRes.ok) throw new Error(`Could not load recordings (HTTP ${recRes.status})`)
      const { recordings } = await recRes.json()
      const recSet = new Set(recordings.map(r => r.filename))

      const valid = entries.filter(([fn]) => recSet.has(fn))
      addLog(`${valid.length} clips have video files on disk`)

      if (valid.length < 10) throw new Error('Not enough labeled clips to train (need ≥ 10)')

      // 2 ── Split clips into train/val BEFORE extraction, so augmented
      // copies are only ever created for training clips.
      const split = splitByGroup(valid)
      if (split.fallback) {
        addLog('⚠ Too few source groups for a grouped split, falling back to random 80/20', 'warn')
      } else {
        addLog(`Group-aware split: ${split.groups} source groups → ${split.train.length} train clips / ${split.val.length} val clips`)
        addLog(`  val per class: ${LABELS.map((l, k) => `${l} ${split.valPerClass[k]}/${split.totalPerClass[k]}`).join(' · ')}`)
      }

      // 3 ── Load MobileNet
      addLog('Loading MobileNet v2…')
      await tf.ready()
      const mobilenet = await loadMobilenet()
      addLog('MobileNet loaded ✓')

      // 4 ── Extract features. Training clips also yield AUG_COPIES augmented
      // variants; validation clips only their original.
      setPhase('extracting')
      const allClips = [
        ...split.train.map(([fn, label]) => [fn, label, 'train']),
        ...split.val.map(([fn, label]) => [fn, label, 'val']),
      ]
      addLog(`Extracting features from ${allClips.length} clips (${FRAMES_PER_CLIP} frames each, ${AUG_COPIES}× augmentation on train)…`)

      const trainFeatures = []
      const trainTargets  = []
      const valFeatures   = []
      const valTargets    = []
      setProgress({ current: 0, total: allClips.length, label: '' })

      for (let i = 0; i < allClips.length; i++) {
        if (stopRef.current) { addLog('Stopped by user.', 'warn'); setPhase('idle'); return }

        const [filename, label, side] = allClips[i]
        setProgress({ current: i + 1, total: allClips.length, label: filename })

        try {
          const video  = await loadVideo(`/recordings/${filename}`)
          const frames = await extractFrames(video, FRAMES_PER_CLIP)
          const variants = side === 'train' ? makeVariants(frames) : [frames]

          for (const variant of variants) {
            // Mean ‖ std of embeddings across frames (see FEATURE_DIM note)
            const embeddings = variant.map(f => embedFrame(mobilenet, f))
            const feat = tf.tidy(() => {
              const stacked = tf.stack(embeddings)              // [N, 1280]
              const { mean, variance } = tf.moments(stacked, 0) // [1280] each
              return tf.concat([mean, variance.sqrt()])         // [2560]
            })
            const arr = await feat.data()

            if (side === 'train') {
              trainFeatures.push(Array.from(arr))
              trainTargets.push(LABELS.indexOf(label))
            } else {
              valFeatures.push(Array.from(arr))
              valTargets.push(LABELS.indexOf(label))
            }
            tf.dispose([...embeddings, feat])
          }
        } catch (err) {
          addLog(`  ⚠ Skipping ${filename}: ${err.message}`, 'warn')
        }
      }

      addLog(`Feature extraction complete: ${trainFeatures.length} train samples (incl. augmented) + ${valFeatures.length} val samples`)
      if (trainFeatures.length === 0 || valFeatures.length === 0) {
        throw new Error('Not enough usable clips after extraction')
      }

      // 4 ── Build + train classifier
      setPhase('training')
      addLog('Building classifier…')

      const xTrain = tf.tensor2d(trainFeatures)                                  // [N, 2560]
      const yTrain = tf.oneHot(tf.tensor1d(trainTargets, 'int32'), LABELS.length) // [N, 5]
      const xVal   = tf.tensor2d(valFeatures)
      const yVal   = tf.oneHot(tf.tensor1d(valTargets, 'int32'), LABELS.length)

      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [FEATURE_DIM], units: 256, activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }) }),
          tf.layers.dropout({ rate: 0.4 }),
          tf.layers.dense({ units: 128, activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }) }),
          tf.layers.dropout({ rate: 0.3 }),
          tf.layers.dense({ units: LABELS.length, activation: 'softmax' }),
        ]
      })

      model.compile({
        optimizer: tf.train.adam(LEARNING_RATE),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      })

      // Inverse-frequency class weights over the TRAINING subset, so
      // under-represented behaviors (standing has ~3× fewer clips than
      // normal) pull their weight in the loss instead of being drowned out.
      const trainClassCount = Array(LABELS.length).fill(0)
      trainTargets.forEach(t => { trainClassCount[t]++ })
      const classWeight = {}
      trainClassCount.forEach((c, k) => {
        classWeight[k] = trainTargets.length / (LABELS.length * Math.max(1, c))
      })
      addLog(`Class weights: ${LABELS.map((l, k) => `${l} ${classWeight[k].toFixed(2)}`).join(' · ')}`)

      addLog(`Training ${trainTargets.length} samples, validating ${valTargets.length}…`)
      addLog(`Architecture: ${FEATURE_DIM} (mean‖std) → 256 → 128 → ${LABELS.length}`)

      const historyLog = []
      // Early stopping with best-weights restore: the last epoch of a long
      // run is usually past the overfitting knee, so keep the weights from
      // the epoch with the lowest validation loss instead.
      let bestValLoss = Infinity
      let bestEpoch   = -1
      let bestWeights = null

      await model.fit(xTrain, yTrain, {
        epochs:          EPOCHS,
        batchSize:       BATCH_SIZE,
        validationData:  [xVal, yVal],
        classWeight,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            const acc    = (logs.acc    ?? logs.accuracy   ?? 0)
            const valAcc = (logs.val_acc ?? logs.val_accuracy ?? 0)
            historyLog.push({ epoch, acc, valAcc })
            if (logs.val_loss < bestValLoss) {
              bestValLoss = logs.val_loss
              bestEpoch   = epoch
              if (bestWeights) bestWeights.forEach(w => w.dispose())
              bestWeights = model.getWeights().map(w => w.clone())
            } else if (epoch - bestEpoch >= PATIENCE) {
              model.stopTraining = true
              addLog(`  Early stop at epoch ${epoch + 1} — val_loss stuck for ${PATIENCE} epochs`)
            }
            if ((epoch + 1) % 5 === 0) {
              addLog(`  Epoch ${epoch + 1}/${EPOCHS}  acc=${(acc * 100).toFixed(1)}%  val_acc=${(valAcc * 100).toFixed(1)}%`)
            }
          },
        },
      })

      if (bestWeights) {
        model.setWeights(bestWeights)
        bestWeights.forEach(w => w.dispose())
        addLog(`Restored best epoch ${bestEpoch + 1}/${historyLog.length} (val_loss ${bestValLoss.toFixed(3)})`)
      }

      const best = historyLog[bestEpoch] ?? historyLog[historyLog.length - 1]
      const finalAcc    = best.acc
      const finalValAcc = best.valAcc

      // 5 ── Confusion matrix on validation set (real clips only, no augmentation)
      const preds  = model.predict(xVal)
      const predIds = Array.from(await preds.argMax(1).data())
      const trueIds = valTargets

      const confMatrix = LABELS.map(() => Array(LABELS.length).fill(0))
      trueIds.forEach((t, i) => { confMatrix[t][predIds[i]]++ })

      tf.dispose([xTrain, yTrain, xVal, yVal, preds])

      setMetrics({ finalAcc, finalValAcc, confMatrix, history: historyLog })
      addLog(`✅ Training complete — train acc ${(finalAcc * 100).toFixed(1)}%  val acc ${(finalValAcc * 100).toFixed(1)}%`, 'success')

      // 6 ── Save model to server
      addLog('Saving model to server…')

      // Serialize manually to send as JSON
      const saveResult = await new Promise((resolve, reject) => {
        model.save(tf.io.withSaveHandler(async (modelArtifacts) => {
          resolve(modelArtifacts)
          return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } }
        }))
      })

      // weightData is an ArrayBuffer — convert to base64 in chunks to avoid call stack overflow
      const bytes = new Uint8Array(saveResult.weightData)
      let binary = ''
      const CHUNK = 8192
      for (let i = 0; i < bytes.length; i += CHUNK) {
        binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
      }
      const weightBase64 = btoa(binary)

      const resp = await fetch('/api/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelTopology: saveResult.modelTopology,
          weightSpecs:   saveResult.weightSpecs,
          weightData:    weightBase64,
          labels:        LABELS,
        }),
      })

      if (!resp.ok) throw new Error(`Save failed: ${resp.statusText}`)

      const info = await resp.json()
      setModelInfo(info)
      addLog('Model saved to server/model/ ✓', 'success')
      setPhase('done')

    } catch (err) {
      addLog(`Error: ${err.message}`, 'error')
      setPhase('error')
    }
  }, [addLog])

  // ── Render ─────────────────────────────────────────────────────────────────
  const isRunning = phase === 'loading-model' || phase === 'extracting' || phase === 'training'

  return (
    <div className="training-studio">
      {/* Header */}
      <div className="ts-header">
        <span className="ts-title">🧠 Training Studio</span>
        <span className="ts-subtitle">Transfer learning via MobileNet v2</span>
      </div>

      {/* Info cards */}
      <div className="ts-info-row">
        {LABELS.map(l => (
          <div key={l} className="ts-label-chip" style={{ borderColor: LABEL_COLOR[l], color: LABEL_COLOR[l] }}>
            {l}
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="ts-controls">
        {!isRunning ? (
          <button className="ts-btn ts-btn-run" onClick={runTraining}>
            ▶ Start Training
          </button>
        ) : (
          <button className="ts-btn ts-btn-stop" onClick={() => { stopRef.current = true }}>
            ■ Stop
          </button>
        )}
      </div>

      {/* Progress bar */}
      {phase === 'extracting' && (
        <div className="ts-progress-wrap">
          <div className="ts-progress-label">
            Extracting features {progress.current}/{progress.total}
          </div>
          <div className="ts-progress-track">
            <div
              className="ts-progress-bar"
              style={{ width: `${(progress.current / progress.total) * 100}%` }}
            />
          </div>
          <div className="ts-progress-file">{progress.label}</div>
        </div>
      )}

      {phase === 'training' && (
        <div className="ts-progress-wrap">
          <div className="ts-progress-label">Training classifier…</div>
        </div>
      )}

      {/* Log */}
      <div className="ts-log">
        {log.map((entry, i) => (
          <div key={i} className={`ts-log-line ts-log-${entry.type}`}>
            <span className="ts-log-time">{entry.time}</span>
            <span>{entry.msg}</span>
          </div>
        ))}
      </div>

      {/* Metrics */}
      {metrics && (
        <div className="ts-metrics">
          <div className="ts-metrics-header">Results</div>
          <div className="ts-metrics-row">
            <div className="ts-metric-card">
              <div className="ts-metric-val">{(metrics.finalAcc * 100).toFixed(1)}%</div>
              <div className="ts-metric-label">Train Accuracy</div>
            </div>
            <div className="ts-metric-card">
              <div className="ts-metric-val">{(metrics.finalValAcc * 100).toFixed(1)}%</div>
              <div className="ts-metric-label">Val Accuracy</div>
            </div>
          </div>

          {/* Confusion matrix */}
          <div className="ts-cm-wrap">
            <div className="ts-cm-title">Confusion Matrix (validation)</div>
            <div className="ts-cm" style={{ gridTemplateColumns: `80px repeat(${LABELS.length}, 1fr)` }}>
              {/* Header row */}
              <div className="ts-cm-corner">true ↓ pred →</div>
              {LABELS.map(l => (
                <div key={l} className="ts-cm-head" style={{ color: LABEL_COLOR[l] }}>{l}</div>
              ))}
              {/* Data rows */}
              {LABELS.map((rowLabel, ri) => (
                <React.Fragment key={rowLabel}>
                  <div className="ts-cm-head" style={{ color: LABEL_COLOR[rowLabel] }}>{rowLabel}</div>
                  {LABELS.map((_cl, ci) => {
                    const val   = metrics.confMatrix[ri][ci]
                    const rowSum = metrics.confMatrix[ri].reduce((a, b) => a + b, 0)
                    const pct   = rowSum > 0 ? val / rowSum : 0
                    const isDiag = ri === ci
                    return (
                      <div
                        key={ci}
                        className={`ts-cm-cell ${isDiag ? 'ts-cm-diag' : ''}`}
                        style={{ opacity: 0.2 + pct * 0.8 }}
                      >
                        {val}
                      </div>
                    )
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      )}

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

        .ts-controls { display: flex; gap: 8px; }

        .ts-btn {
          padding: 8px 20px;
          border-radius: var(--radius);
          font-size: 13px;
          font-family: var(--font-display);
          cursor: pointer;
          transition: opacity 0.15s;
        }
        .ts-btn:hover { opacity: 0.85; }
        .ts-btn-run {
          background: var(--accent);
          color: #000;
        }
        .ts-btn-stop {
          background: var(--red, #ff4444);
          color: #fff;
        }

        .ts-progress-wrap {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 10px 14px;
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .ts-progress-label { font-size: 12px; color: var(--text-secondary); }
        .ts-progress-file  { font-size: 10px; color: var(--text-muted); font-family: monospace; }
        .ts-progress-track {
          height: 4px;
          background: var(--border);
          border-radius: 2px;
          overflow: hidden;
        }
        .ts-progress-bar {
          height: 100%;
          background: var(--accent);
          border-radius: 2px;
          transition: width 0.3s;
        }

        .ts-log {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 10px 14px;
          max-height: 220px;
          overflow-y: auto;
          font-family: monospace;
          font-size: 11px;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .ts-log-line { display: flex; gap: 8px; }
        .ts-log-time { color: var(--text-muted); flex-shrink: 0; }
        .ts-log-info    { color: var(--text-secondary); }
        .ts-log-success { color: #7dff7d; }
        .ts-log-warn    { color: #ffd264; }
        .ts-log-error   { color: #ff6b6b; }

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
        .ts-metrics-row { display: flex; gap: 12px; }
        .ts-metric-card {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 10px 16px;
          text-align: center;
          min-width: 100px;
        }
        .ts-metric-val {
          font-size: 24px;
          font-family: var(--font-display);
          color: var(--accent);
        }
        .ts-metric-label { font-size: 10px; color: var(--text-muted); margin-top: 2px; }

        .ts-cm-wrap { display: flex; flex-direction: column; gap: 6px; }
        .ts-cm-title { font-size: 11px; color: var(--text-muted); }
        .ts-cm {
          display: grid;
          gap: 2px;
        }
        .ts-cm-corner {
          font-size: 8px;
          color: var(--text-muted);
          display: flex;
          align-items: flex-end;
          padding-bottom: 4px;
        }
        .ts-cm-head {
          font-size: 9px;
          text-align: center;
          padding: 3px 2px;
          font-family: var(--font-display);
        }
        .ts-cm-cell {
          background: #88aaff;
          text-align: center;
          font-size: 11px;
          padding: 4px 2px;
          border-radius: 2px;
          color: #fff;
          font-family: var(--font-display);
        }
        .ts-cm-diag { background: #7dff7d; color: #000; }
      `}</style>
    </div>
  )
}
