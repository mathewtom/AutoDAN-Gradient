#!/usr/bin/env bash
# Campaign D — descriptive-target leak suite (D1 tool description, D2 approval
# matrix). Sequenced via the adaptive orchestrator with the post-Campaign-C
# defaults (steps=500, plateau window=125). Designed against the patterns
# documented in docs/basin_patterns.html.

set +e
cd /Users/mainuser/Documents/RAG/AutoDAN-Gradient

YAMLS=(
    "campaigns/campaign_d1_tool_description.yaml"
    "campaigns/campaign_d2_approval_matrix.yaml"
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
echo "===== CAMPAIGN D FULLY COMPLETE ====="
date
