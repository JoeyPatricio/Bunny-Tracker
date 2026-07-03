import { useState, useEffect, useRef, useCallback } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as mobilenetModule from '@tensorflow-models/mobilenet'

// 1.5s matches the frame spacing the classifier was trained on (8 frames
// sampled evenly across a ~12s clip in TrainingStudio).
const INFERENCE_INTERVAL_MS = 1500
// Rolling window for embedding averaging — must match FRAMES_PER_CLIP in
// TrainingStudio: the classifier expects the MEAN of this many frame
// embeddings, not a single-frame embedding.
const EMBED_WINDOW = 8
const LABEL_COLOR = {
  grooming: '#dc82ff',
  normal:   '#88aaff',
  standing: '#ff9f3c',
  yawn:     '#ffd264',
  zoomies:  '#7dff7d',
}

export function useInference({ videoRef, isActive, enabled }) {
  const [status, setStatus]         = useState('idle') // idle | loading | ready | error
  const [prediction, setPrediction] = useState(null)   // { label, confidence, color }
  const [modelExists, setModelExists] = useState(null) // null = unchecked

  const mobilenetRef   = useRef(null)
  const classifierRef  = useRef(null)
  const labelsRef      = useRef([])
  const intervalRef    = useRef(null)
  const canvasRef      = useRef(null)
  const embedWindowRef = useRef([]) // rolling single-frame embeddings (Float32Array each)

  // Check if a trained model is saved on the server
  useEffect(() => {
    fetch('/api/model')
      .then(r => r.json())
      .then(d => setModelExists(d.exists))
      .catch(() => setModelExists(false))
  }, [])

  const loadModels = useCallback(async () => {
    if (status === 'loading' || status === 'ready') return
    setStatus('loading')
    try {
      await tf.ready()

      // Load MobileNet backbone (same as training) — self-hosted copy first
      // (CSP blocks tfhub.dev; run server/scripts/download-mobilenet.mjs),
      // CDN as fallback. inputRange [0,1] is required with modelUrl.
      if (!mobilenetRef.current) {
        try {
          mobilenetRef.current = await mobilenetModule.load({
            version: 2, alpha: 1.0,
            modelUrl: '/model/mobilenet/model.json',
            inputRange: [0, 1],
          })
        } catch {
          mobilenetRef.current = await mobilenetModule.load({ version: 2, alpha: 1.0 })
        }
      }

      // Load the trained classifier from server
      classifierRef.current = await tf.loadLayersModel('/model/model.json')

      // Load label map
      const labelsRes = await fetch('/model/labels.json')
      labelsRef.current = await labelsRes.json()

      // Reusable 224×224 canvas for frame capture
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas')
        canvasRef.current.width  = 224
        canvasRef.current.height = 224
      }

      setStatus('ready')
    } catch (err) {
      console.error('Inference model load failed:', err)
      setStatus('error')
    }
  }, [status])

  const runOnce = useCallback(() => {
    const video = videoRef.current
    if (!video || video.readyState < 2) return
    if (!mobilenetRef.current || !classifierRef.current) return

    const ctx = canvasRef.current.getContext('2d')
    ctx.drawImage(video, 0, 0, 224, 224)

    // Single-frame embedding → rolling mean over the last EMBED_WINDOW frames.
    // The classifier was trained on the mean of 8 frame embeddings per clip
    // (TrainingStudio), so inference must aggregate the same way.
    const embedding = tf.tidy(() =>
      mobilenetRef.current
        .infer(tf.browser.fromPixels(canvasRef.current), true)      // [1, 1280]
        .dataSync()                                                 // Float32Array [1280]
    )
    const window = embedWindowRef.current
    window.push(embedding)
    if (window.length > EMBED_WINDOW) window.shift()

    // Build mean ‖ std features (motion signal) for a retrained model, or
    // mean only for an older 1280-input model — match what's loaded.
    const dim  = embedding.length
    const withStd = classifierRef.current.inputs[0].shape[1] === dim * 2
    const feat = new Float32Array(withStd ? dim * 2 : dim)
    for (const emb of window) {
      for (let i = 0; i < dim; i++) feat[i] += emb[i]
    }
    for (let i = 0; i < dim; i++) feat[i] /= window.length
    if (withStd) {
      for (const emb of window) {
        for (let i = 0; i < dim; i++) {
          const d = emb[i] - feat[i]
          feat[dim + i] += d * d
        }
      }
      for (let i = 0; i < dim; i++) feat[dim + i] = Math.sqrt(feat[dim + i] / window.length)
    }

    tf.tidy(() => {
      const logits    = classifierRef.current.predict(tf.tensor2d(feat, [1, feat.length])) // [1, 5]
      const probs     = logits.squeeze()                             // [5]
      const predIdx   = probs.argMax().dataSync()[0]
      const confidence = probs.dataSync()[predIdx]

      const label = labelsRef.current[predIdx] ?? `class_${predIdx}`
      setPrediction({
        label,
        confidence: Math.round(confidence * 100),
        color: LABEL_COLOR[label] ?? '#ffffff',
      })
    })
  }, [videoRef])

  // Start / stop inference interval when enabled + ready + active
  useEffect(() => {
    if (enabled && status === 'ready' && isActive) {
      runOnce() // immediate first reading
      intervalRef.current = setInterval(runOnce, INFERENCE_INTERVAL_MS)
    } else {
      clearInterval(intervalRef.current)
      intervalRef.current = null
      embedWindowRef.current = [] // stale frames shouldn't blend into the next session
      if (!isActive || !enabled) setPrediction(null)
    }
    return () => clearInterval(intervalRef.current)
  }, [enabled, status, isActive, runOnce])

  // Auto-load when enabled
  useEffect(() => {
    if (enabled && modelExists && status === 'idle') {
      loadModels()
    }
  }, [enabled, modelExists, status, loadModels])

  return { status, prediction, modelExists, loadModels }
}
