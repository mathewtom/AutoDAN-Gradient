#!/usr/bin/env bash
# Campaign C extension — meta-reference targets C3, C4, C8, C5, C6.
# Each is a separate adaptive-orchestrator run. Fires sequentially in
# priority order. Does NOT auto-start the SecureRAG-Agent or run
# transfer tests — that's a manual step after the GCG runs settle.

set +e
cd /Users/mainuser/Documents/RAG/AutoDAN-Gradient

YAMLS=(
    "campaigns/campaign_c3_busy_placeholder.yaml"
    "campaigns/campaign_c4_identity_runtime.yaml"
    "campaigns/campaign_c8_embedded_instructions.yaml"
    "campaigns/campaign_c5_tool_hop_cap.yaml"
    "campaigns/campaign_c6_operational_guidelines.yaml"
)

for cfg in "${YAMLS[@]}"; do
    echo
    echo "===== ADAPTIVE CAMPAIGN: $cfg ====="
    date
    uv run python scripts/run_adaptive_campaign.py --config "$cfg"
    rc=$?
    echo
    echo "===== DONE: $cfg (exit $rc) ====="
    date
done

echo
echo "===== CAMPAIGN C EXTENSION FULLY COMPLETE ====="
date
