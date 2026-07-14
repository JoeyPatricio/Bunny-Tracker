/**
 * Windows install bug in @tensorflow/tfjs-node: the native binding looks for
 * tensorflow.dll next to itself under lib/ (napi-v8 on Node 22), but npm
 * install leaves the DLL in deps/lib/, so require() fails with
 * ERR_DLOPEN_FAILED and the agent silently runs on the pure-JS backend.
 * Copy it next to the binding.
 * Runs as postinstall; no-op on non-Windows and when already fixed.
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

if (process.platform === 'win32') {
  const pkg = path.join(path.dirname(fileURLToPath(import.meta.url)),
    '..', 'node_modules', '@tensorflow', 'tfjs-node')
  const src = path.join(pkg, 'deps', 'lib', 'tensorflow.dll')
  const libDir = path.join(pkg, 'lib')
  if (fs.existsSync(src) && fs.existsSync(libDir)) {
    for (const d of fs.readdirSync(libDir).filter(n => n.startsWith('napi-'))) {
      const dest = path.join(libDir, d, 'tensorflow.dll')
      if (!fs.existsSync(dest)) {
        fs.copyFileSync(src, dest)
        console.log(`tfjs-node: copied tensorflow.dll into lib/${d}`)
      }
    }
  }
}
