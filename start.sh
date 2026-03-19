#!/bin/bash
# EDAgent Launcher — starts backend API + OpenWebUI frontend

cd "$(dirname "$0")"

# Kill any existing instances
pkill -f "app.py" 2>/dev/null
pkill -f "open-webui serve" 2>/dev/null
sleep 1

# Start EDAgent backend API (port 5001)
echo "Starting EDAgent API on port 5001..."
/opt/anaconda3/bin/python3 app.py > /tmp/edagent_api.log 2>&1 &
echo "EDAgent API PID: $!"

# Wait for API to be ready
sleep 3

# Start OpenWebUI frontend (port 3000)
echo "Starting OpenWebUI on port 3000..."
OPENAI_API_BASE_URL=http://localhost:5001/v1 \
OPENAI_API_KEY=sk-dummy \
/opt/anaconda3/bin/open-webui serve --port 3000 > /tmp/edagent_ui.log 2>&1 &
echo "OpenWebUI PID: $!"

echo ""
echo "======================================"
echo "  EDAgent is starting up..."
echo "  Open http://localhost:3000 in your browser"
echo "  Login: admin@edagent.local / edagent123"
echo "  Select model: ed-agent-react"
echo "======================================"
