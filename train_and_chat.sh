#!/bin/bash

LOCKFILE="/tmp/bash_ai.lock"

if [ -f "$LOCKFILE" ]; then
    echo "🐺 Another instance of Bash AI training is already running!"
    exit 1
else
    touch "$LOCKFILE"
fi

trap "rm -f $LOCKFILE" EXIT

# Activate virtual environment
echo "🐺 Activating virtual environment..."
source backend/venv/bin/activate

# Train the AI model
echo "🐺 Training Bash AI... Hold tight, pup!"
PYTHONPATH=backend python backend/ai_engine/train.py

# Notify user
if [ $? -eq 0 ]; then
    echo "🎉 Bash AI training complete! Time to test him out, pup!"
else
    echo "❌ Something went wrong during training."
    exit 1
fi

# Start the server in the background
echo "🚀 Starting Bash AI server..."
PYTHONPATH=backend python backend/api/server.py &
SERVER_PID=$!

# Server readiness check
MAX_WAIT=15
COUNTER=0
until curl -s http://localhost:8000/docs > /dev/null; do
    sleep 1
    COUNTER=$((COUNTER + 1))
    if [ $COUNTER -ge $MAX_WAIT ]; then
        echo "❌ Bash AI server failed to start."
        kill $SERVER_PID
        exit 1
    fi
done

echo "✅ Bash AI server is up and running!"

# Prompt user loop
while true; do
    read -p "💬 Enter a message for Bash AI (or type 'exit' to quit): " user_message

    if [ "$user_message" == "exit" ]; then
        echo "👋 Shutting down Bash AI server..."
        kill $SERVER_PID
        wait $SERVER_PID
        echo "👋 Goodbye, pup!"
        break
    fi

    # Send message via curl
    response=$(curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"text\": \"$user_message\"}")

    echo "🐺 Bash AI: $(echo "$response" | jq -r '.response')"
done
