/**
 * PM2 process definitions for BunnyCam.
 * Start everything with:  pm2 start ecosystem.config.cjs
 * This declarative list is the source of truth — more reliable than pm2 save.
 */
const path = require('path')
const SERVER = path.join(__dirname, 'server')

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
      script: 'index.js',
      cwd: SERVER,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 3000,
    },
    {
      name: 'bunnycam-agent',
      script: 'agent/capture.mjs',
      cwd: SERVER,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
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
        N8N_RESTRICT_FILE_ACCESS_TO: 'C:\\Users\\joepa\\Desktop\\PetCam\\server\\recordings',
        GENERIC_TIMEZONE: 'America/Los_Angeles',
      },
    },
  ],
}
