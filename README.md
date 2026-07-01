# RAG Driving License Telegram Bot

Telegram bot that accepts driving theory screenshots, extracts the question/options with a vision LLM, and can optionally answer using a local LlamaIndex RAG index built from DGT driving documents.

## Deploy on Linode/Akamai Ubuntu 24.04

Recommended first deploy mode: `BOT_MODE=polling`. Polling does not need a domain, HTTPS, Nginx, or Telegram webhook setup.

### 1. Create the server

Use an Ubuntu 24.04 Linode with 4 GB RAM.

Log in:

```bash
ssh root@YOUR_SERVER_IP
```

Install system packages:

```bash
apt update
apt upgrade -y
apt install -y python3 python3-venv python3-pip git rsync
```

Create a service user:

```bash
adduser --system --group --home /opt/rag-driving-license botuser
```

### 2. Upload or clone the project

Option A, clone from Git:

```bash
git clone YOUR_REPO_URL /opt/rag-driving-license
chown -R botuser:botuser /opt/rag-driving-license
```

Option B, upload from your Mac:

```bash
rsync -av --exclude venv --exclude .env --exclude __pycache__ ./ root@YOUR_SERVER_IP:/opt/rag-driving-license/
ssh root@YOUR_SERVER_IP 'chown -R botuser:botuser /opt/rag-driving-license'
```

Important: keep the `storage/` directory on the server if `PREBUILT_INDEX=1` and `ENABLE_RAG=1`.

### 3. Install Python dependencies

```bash
cd /opt/rag-driving-license
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
chown -R botuser:botuser /opt/rag-driving-license
```

### 4. Configure environment

```bash
cd /opt/rag-driving-license
cp .env.example .env
nano .env
```

Minimum useful polling configuration:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
BOT_MODE=polling

PARSE_PROVIDER=xai
XAI_API_KEY=your_xai_key_here
PARSE_MODEL=grok-4

ENABLE_RAG=0
PREBUILT_INDEX=1

LIMIT_PER_DAY=10
LIMIT_STORE=./user_limits.sqlite3
RESET_PRICE_STARS=10
RESET_INCREMENT=10
```

For RAG answers, also set:

```env
ENABLE_RAG=1
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=grok
GROK_MODEL=grok-4-0709
RAG_TOP_K=2
RAG_MIN_SCORE=0.3
```

Fix permissions:

```bash
chown botuser:botuser /opt/rag-driving-license/.env
chmod 600 /opt/rag-driving-license/.env
```

### 5. Test manually

```bash
cd /opt/rag-driving-license
sudo -u botuser ./venv/bin/python bot.py
```

Send `/start` to the bot in Telegram. Stop the manual run with `Ctrl+C`.

### 6. Install systemd service

```bash
cp /opt/rag-driving-license/deploy/rag-driving-bot.service /etc/systemd/system/rag-driving-bot.service
systemctl daemon-reload
systemctl enable rag-driving-bot
systemctl start rag-driving-bot
```

Check status and logs:

```bash
systemctl status rag-driving-bot
journalctl -u rag-driving-bot -f
```

Restart after code or env changes:

```bash
systemctl restart rag-driving-bot
```

### 7. Updating the bot

If using Git:

```bash
cd /opt/rag-driving-license
git pull
./venv/bin/pip install -r requirements.txt
chown -R botuser:botuser /opt/rag-driving-license
systemctl restart rag-driving-bot
```

If uploading from your Mac:

```bash
rsync -av --exclude venv --exclude .env --exclude __pycache__ ./ root@YOUR_SERVER_IP:/opt/rag-driving-license/
ssh root@YOUR_SERVER_IP 'cd /opt/rag-driving-license && ./venv/bin/pip install -r requirements.txt && chown -R botuser:botuser . && systemctl restart rag-driving-bot'
```

## Webhook mode later

Use webhook mode only after polling works. You will need a domain, HTTPS, and a public URL:

```env
BOT_MODE=webhook
WEBHOOK_URL=https://your-domain.example
WEBHOOK_PATH=/webhook
PORT=8080
```

Then put Nginx/Caddy in front of the bot and proxy `WEBHOOK_PATH` to local port `8080`.
