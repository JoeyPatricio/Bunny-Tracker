/**
 * PM2 process definitions for BunnyCam.
 * Start everything with:  pm2 start ecosystem.config.cjs
 * This declarative list is the source of truth — more reliable than pm2 save.
 *
 * autorestart is ON everywhere so a genuine crash self-heals. The catch:
 * with autorestart on, pm2 treats a manual `pm2 stop`/kill the same as a
 * crash and immediately relaunches it — `pm2 stop` alone won't actually
 * stop the app. The real kill switch is `pm2 kill` (tears down the whole
 * pm2 daemon, so there's nothing left to relaunch); the console window
 * opened by `C:\Users\joepa\.pm2\start-bunnycam.bat` at login offers this
 * via a "press K" prompt instead of closing.
 */
const path = require('path')
const fs = require('fs')
const { execFileSync } = require('child_process')

const WINDOWS = process.platform === 'win32'
const SERVER_PY = path.join(__dirname, 'server_py')
// The runtime venv's own interpreter directly, not `interpreter: 'python3'` —
// that assumes a PATH entry this Windows box does not guarantee. The venv
// layout differs by platform: Scripts/python.exe vs. bin/python.
const PYTHON = WINDOWS
  ? path.join(SERVER_PY, '.venv', 'Scripts', 'python.exe')
  : path.join(SERVER_PY, '.venv', 'bin', 'python')

// Resolve a binary that lives on PATH on Linux but at a fixed install path on
// Windows. Returns null when it isn't installed, so the app list can drop it
// rather than hand pm2 a script it will crash-loop on to max_restarts.
function resolveBin(envOverride, windowsPath, unixName) {
  if (envOverride) return envOverride
  if (WINDOWS) return fs.existsSync(windowsPath) ? windowsPath : null
  try {
    // stderr ignored: `which` prints a not-found line, and a missing
    // optional binary is a normal dev-box state, not something to report.
    return execFileSync('which', [unixName],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim() || null
  } catch {
    return null
  }
}

// cloudflared. Override with CLOUDFLARED_PATH.
const CLOUDFLARED = resolveBin(
  process.env.CLOUDFLARED_PATH,
  'C:\\Program Files (x86)\\cloudflared\\cloudflared.exe',
  'cloudflared')

// n8n entry point (global npm install). Override with N8N_BIN.
const N8N_BIN = resolveBin(
  process.env.N8N_BIN,
  'C:\\Users\\joepa\\AppData\\Local\\Packages\\Claude_pzs8sxrjxfjjc\\LocalCache\\Roaming\\npm\\node_modules\\n8n\\bin\\n8n',
  'n8n')

const RECORDINGS = path.join(__dirname, 'server', 'recordings')

const apps = [
    {
      name: 'bunnycam-server',
      script: PYTHON,
      args: '-m uvicorn app.main:app --host 0.0.0.0 --port 3001',
      cwd: SERVER_PY,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 3000,
      env: { PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1', PY_PORT: '3001' },
    },
    {
      name: 'bunnycam-agent',
      script: PYTHON,
      args: '-m agent.capture',
      cwd: SERVER_PY,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      env: { PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
    },
    {
      name: 'bunnycam-tunnel',
      script: CLOUDFLARED,
      args: 'tunnel run bunnycam',
      interpreter: 'none',
      autorestart: true,
      max_restarts: 20,
      restart_delay: 3000,
    },
    {
      name: 'bunnycam-n8n',
      script: N8N_BIN,
      interpreter: 'node',
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        N8N_PORT: '5678',
        N8N_LISTEN_ADDRESS: '127.0.0.1', // localhost-only — not exposed
        N8N_HOST: 'localhost',
        N8N_SECURE_COOKIE: 'false',       // allow login over local http
        N8N_RUNNERS_ENABLED: 'true',
        N8N_DIAGNOSTICS_ENABLED: 'false', // disable telemetry
        // Allow the Read/Write File node to reach the clips (2.x sandboxes it
        // to .n8n-files by default).
        N8N_RESTRICT_FILE_ACCESS_TO: RECORDINGS,
        GENERIC_TIMEZONE: 'America/Los_Angeles',
      },
    },
]

// Drop the processes whose binary this machine doesn't have (cloudflared and
// n8n are absent on a plain dev box); server + agent always run.
module.exports = { apps: apps.filter((a) => a.script) }
