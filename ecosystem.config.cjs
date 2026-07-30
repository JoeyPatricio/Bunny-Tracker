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
const SERVER_PY = path.join(__dirname, 'server_py')
// The runtime venv's own python.exe directly, not `interpreter: 'python3'` —
// that assumes a PATH entry this Windows box does not guarantee.
const PYTHON = path.join(SERVER_PY, '.venv', 'Scripts', 'python.exe')

// cloudflared location (Windows install path). Override with CLOUDFLARED_PATH.
const CLOUDFLARED = process.env.CLOUDFLARED_PATH ||
  'C:\\Program Files (x86)\\cloudflared\\cloudflared.exe'

// n8n entry point (global npm install). Override with N8N_BIN.
const N8N_BIN = process.env.N8N_BIN ||
  'C:\\Users\\joepa\\AppData\\Local\\Packages\\Claude_pzs8sxrjxfjjc\\LocalCache\\Roaming\\npm\\node_modules\\n8n\\bin\\n8n'

module.exports = {
  apps: [
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
        N8N_RESTRICT_FILE_ACCESS_TO: 'C:\\Users\\joepa\\Desktop\\BunnyTracker\\server\\recordings',
        GENERIC_TIMEZONE: 'America/Los_Angeles',
      },
    },
  ],
}
