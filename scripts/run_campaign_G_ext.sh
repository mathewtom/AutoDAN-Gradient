#!/usr/bin/env bash
# Campaign G — confidentiality-clause self-leak suite (G1 "Treat these
# instructions", G2 "any internal configuration"). Sequenced via the
# adaptive orchestrator with post-Campaign-D defaults (steps=500,
# plateau window=125, fallback plateau window=70).

set +e
cd /Users/mainuser/Documents/RAG/AutoDAN-Gradient

YAMLS=(
    "campaigns/campaign_g1_treat_instructions.yaml"
    "campaigns/campaign_g2_internal_configuration.yaml"
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
echo "===== CAMPAIGN G FULLY COMPLETE ====="
date
