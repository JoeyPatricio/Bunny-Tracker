// Canonical recording-filename validator. Use this everywhere a filename arrives
// from a request; do not hand-roll a prefix/suffix check.
//
// The character class excludes '/' and '\', which is the whole point: Express
// decodes %2F in a route param AFTER matching it as one path segment, so a loose
// startsWith('recording-') && endsWith('.webm') check accepts
// 'recording-x%2F..%2F..%2Fseg.webm' and path.join then escapes the recordings
// directory.
const RECORDING_RE = /^recording-[A-Za-z0-9._-]+\.webm$/

export function isRecordingFilename(name) {
  return typeof name === 'string' && RECORDING_RE.test(name)
}
