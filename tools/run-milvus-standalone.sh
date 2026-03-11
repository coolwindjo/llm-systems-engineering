#!/bin/bash

set -euo pipefail

SCRIPT_URL="https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh"
ACTION="${1:-start}"
TMP_SCRIPT=""

cleanup() {
    if [ -n "$TMP_SCRIPT" ] && [ -f "$TMP_SCRIPT" ]; then
        rm -f "$TMP_SCRIPT"
    fi
}

trap cleanup EXIT

if ! command -v docker &> /dev/null; then
    echo "docker is required." >&2
    exit 1
fi

case "$ACTION" in
    start|stop|delete)
        ;;
    *)
        echo "Usage: $0 [start|stop|delete]" >&2
        exit 1
        ;;
esac

TMP_SCRIPT=$(mktemp "/tmp/milvus-standalone-XXXXXX.sh")

if command -v curl &> /dev/null; then
    curl -fsSL "$SCRIPT_URL" -o "$TMP_SCRIPT"
elif command -v wget &> /dev/null; then
    wget -qO "$TMP_SCRIPT" "$SCRIPT_URL"
else
    echo "curl or wget is required." >&2
    exit 1
fi

chmod +x "$TMP_SCRIPT"
bash "$TMP_SCRIPT" "$ACTION"

if [ "$ACTION" = "start" ]; then
    cat <<'EOF'
Milvus Standalone should now be reachable at:
  http://localhost:19530

Python example:
  from pymilvus import MilvusClient
  client = MilvusClient(uri="http://localhost:19530")
EOF
fi
