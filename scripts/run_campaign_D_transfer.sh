#!/usr/bin/env bash
# Transfer-test all non-abandoned Campaign D runs against the live
# SecureRAG-Agent. Discovers eligible JSONLs at runtime (any D1/D2
# JSONL without an `"abandoned": true` marker — covers productive,
# plateau-aborted, and fallback runs).

set +e
cd /Users/mainuser/Documents/RAG/AutoDAN-Gradient

AUDIT="/Users/mainuser/Documents/RAG/SecureRAG-Agent/logs"
URL="http://127.0.0.1:8000"
TS=$(date +%Y%m%d_%H%M)
OUT_DIR="results/scanner_evasion"

declare -A TARGETS
TARGETS[D1]="semantic search over the Meridian knowledge base"
TARGETS[D2]="per the 2026 approval matrix"

echo "===== DISCOVERING ELIGIBLE D JSONLs ====="
ELIGIBLE=()
for f in $OUT_DIR/D[12]_D[12]_*.jsonl; do
    [ -e "$f" ] || continue
    KIND=$(python3 -c "
import json
recs=[json.loads(l) for l in open('$f') if l.strip()]
print('abandoned' if any(r.get('abandoned') for r in recs) else 'eligible')")
    if [ "$KIND" = "eligible" ]; then
        ELIGIBLE+=("$f")
        echo "  + $(basename "$f")"
    fi
done
echo "Found ${#ELIGIBLE[@]} eligible runs"

if [ ${#ELIGIBLE[@]} -eq 0 ]; then
    echo "No eligible JSONLs — nothing to transfer-test"
    exit 0
fi

echo
echo "===== STARTING SECURERAG-AGENT ====="
date
cd /Users/mainuser/Documents/RAG/SecureRAG-Agent
uv run uvicorn securerag_agent.api:app --host 127.0.0.1 --port 8000 > /tmp/securerag_agent.log 2>&1 &
AGENT_PID=$!
cd /Users/mainuser/Documents/RAG/AutoDAN-Gradient
echo "agent pid: $AGENT_PID; waiting for /health..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q ok; then
        echo "agent up after ${i} attempts"
        break
    fi
    sleep 2
done

for f in "${ELIGIBLE[@]}"; do
    BASENAME=$(basename "$f" .jsonl)
    CAMPAIGN_ID=$(echo "$BASENAME" | grep -oE '^D[12]')
    TARGET="${TARGETS[$CAMPAIGN_ID]}"
    echo
    echo "===== TRANSFER ${BASENAME} (target=\"$TARGET\") ====="
    date
    uv run python scripts/transfer_test_top5.py \
        --campaign-jsonl "$f" \
        --target-string "$TARGET" \
        --audit-dir "$AUDIT" --service-url "$URL" \
        --out "$OUT_DIR/transfer_${BASENAME}_${TS}"
done

echo
echo "===== SHUTTING DOWN AGENT + 70B ====="
date
kill $AGENT_PID 2>/dev/null
sleep 3
ollama stop llama3.3:70b 2>&1
echo
echo "===== TRANSFER TESTS COMPLETE ====="
date
