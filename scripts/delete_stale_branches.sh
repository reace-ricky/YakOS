#!/usr/bin/env bash
# delete_stale_branches.sh — prune all merged/stale branches
#
# Keeps: main, copilot/remove-lineup-vs-actuals-section
# Deletes: all other ~145 copilot/*, feature/*, fix/*, refactor/*, ui/* branches
#
# Usage:
#   chmod +x scripts/delete_stale_branches.sh
#   ./scripts/delete_stale_branches.sh

set -euo pipefail

BRANCHES=(
  copilot/add-backtest-ticket-format
  copilot/add-calibration-tab-functionality
  copilot/add-dk-contest-ingestion
  copilot/add-dvp-csv-upload-feature
  copilot/add-exposure-heatmap-indicator
  copilot/add-exposure-review-heatmap
  copilot/add-exposure-review-panel
  copilot/add-external-ownership-ingestion
  copilot/add-kpi-analysis-tools
  copilot/add-kpi-boxes-calibration-page
  copilot/add-lineup-boom-bust-ranking
  copilot/add-lineup-card-display
  copilot/add-manual-injury-overrides
  copilot/add-manual-tag-overrides
  copilot/add-optimizer-rules-builder
  copilot/add-player-projections-table
  copilot/add-player-table-for-sim-calibration
  copilot/add-pool-size-gauge
  copilot/add-proj-pts-own-percent
  copilot/add-projection-functions-to-app
  copilot/add-projections-to-calibration-lab
  copilot/add-rci-gauge-system
  copilot/add-rickys-calibration-lab
  copilot/add-rickys-edge-analysis-template
  copilot/add-salary-history-client
  copilot/add-sim-eligible-flag
  copilot/add-sims-calibration-functionality
  copilot/auto-detect-not-with-pairs
  copilot/build-alert-validation-pipeline
  copilot/build-foundation-dataset
  copilot/check-task-status
  copilot/clean-up-calibration-kpi-dashboard
  copilot/consolidate-lineup-builder-csv
  copilot/consolidate-slate-hub
  copilot/debug-ownership-merge
  copilot/default-to-est-and-cleanup
  copilot/define-contest-goal-summary-helper
  copilot/define-contest-smash-bust-thresholds
  copilot/discuss-api-usage-for-calibration
  copilot/enhance-sim-module-anomaly-detection
  copilot/extend-contest-presets
  copilot/extend-ricky-edge-state
  copilot/fix-app-access-issues
  copilot/fix-app-launch-error
  copilot/fix-app-launch-error-again
  copilot/fix-calibration-actuals-loading
  copilot/fix-calibration-actuals-loading-again
  copilot/fix-calibration-api-key-prompt
  copilot/fix-calibration-tab-display
  copilot/fix-cancellation-reason-logic
  copilot/fix-cancelled-request-issue
  copilot/fix-contest-type-thresholds
  copilot/fix-custom-lineup-builder-slate
  copilot/fix-dashboard-projection-issue
  copilot/fix-dff-fallback-logic
  copilot/fix-duplicated-players-in-pool
  copilot/fix-import-error-in-app
  copilot/fix-injury-cascade-issue
  copilot/fix-input-validation-error
  copilot/fix-launch-error
  copilot/fix-memory-loop-issue
  copilot/fix-ownership-display-in-slate-hub
  copilot/fix-ownership-merge-bug
  copilot/fix-ownership-pipeline-external-data
  copilot/fix-player-pool-date-bug
  copilot/fix-position-extraction-bug
  copilot/fix-projection-source-issues
  copilot/fix-sim-error-argument
  copilot/fix-sim-module-error
  copilot/fix-sims-module-error
  copilot/fix-sims-module-injury-issue
  copilot/fix-sims-module-injury-logic
  copilot/fix-slate-hub-date-default
  copilot/fix-smash-bust-definitions
  copilot/fix-stack-alert-projections
  copilot/fix-tank01-actuals-api
  copilot/fix-tank01-api-error
  copilot/fix-valueerror-sim-player-pool
  copilot/get-status-update
  copilot/group-and-dedupe-players
  copilot/hide-dk-contest-selection
  copilot/identify-ownership-column
  copilot/include-boom-bust-metrics
  copilot/instrument-simulations-pool-check
  copilot/introduce-global-slate-state
  copilot/investigate-session-cancellations
  copilot/make-slate-room-alerts-robust
  copilot/na
  copilot/prepare-sims-table-func
  copilot/qc-audit-column-naming
  copilot/qc-audit-column-naming-standardization
  copilot/re-engineer-fetch-api-button
  copilot/read-memory-loop
  copilot/redistribute-projected-minutes
  copilot/refactor-calibration-kpi-strip
  copilot/refactor-dynamic-smash-bust
  copilot/refactor-simulations-pipeline
  copilot/remove-contest-columns-dashboard
  copilot/remove-extra-columns-and-controls
  copilot/remove-metrics-headers-and-circles
  copilot/remove-projection-model-selectbox
  copilot/remove-unnecessary-elements
  copilot/reorder-pages-and-move-slate-context
  copilot/replace-internal-ownership-with-external-pown
  copilot/resolve-pull-request-conflicts
  copilot/restructure-app-role-tabs
  copilot/review-canceled-sessions
  copilot/rewrite-edge-share-page
  copilot/scaffold-streamlit-app
  copilot/streamline-calibration-workflow
  copilot/test-sim-module-backtest
  copilot/trace-fetch-pool-calibration
  copilot/trace-player-pool-issues
  copilot/train-and-commit-models
  copilot/understanding-sims-module
  copilot/unify-source-for-anomalies
  copilot/update-auto-tagging-confidence
  copilot/update-build-suggestion-status
  copilot/update-calibration-tab-options
  copilot/update-contest-type-dropdown
  copilot/update-dk-lobby-api-flow
  copilot/update-lab-injury-cascade-ownership
  copilot/update-player-pool-load
  copilot/update-pool-load-date-dependency
  copilot/update-projection-system
  copilot/update-runtime-file
  copilot/update-slate-room-code
  copilot/update-slate-room-layout
  copilot/update-smash-bust-thresholds
  copilot/update-streamlit-app-rotogrinders
  copilot/verify-player-pool-wiring
  copilot/verify-wire-projection-uploads
  copilot/what-is-next-task
  copilot/whats-next-reflection
  copilot/wire-calibration-loop
  copilot/wire-slate-hub-to-slatestate
  feature/field-sim-ownership
  feature/slate-picker-contest-config
  fix/dff-fallback-pool-loading
  fix/historical-pool-loading
  fix/right-angle-ricky-optimizer-tab
  fix/sim-sandbox-duplicates-and-pga-optimizer
  fix/slate-hub-historical-dropdowns
  refactor/dk-pool-tank01-stats
  ui/consolidate-lab-page
)

BATCH_SIZE=20
FAILED=()

echo "Deleting ${#BRANCHES[@]} stale branches from origin (${BATCH_SIZE} at a time)..."

for (( i=0; i<${#BRANCHES[@]}; i+=BATCH_SIZE )); do
  batch=("${BRANCHES[@]:$i:$BATCH_SIZE}")
  echo "  batch $((i/BATCH_SIZE + 1)): ${batch[*]}"
  git push origin --delete "${batch[@]}" || {
    echo "  WARNING: some deletions in this batch failed; continuing..."
    FAILED+=("${batch[@]}")
  }
done

echo ""
echo "Done."
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "The following branches could not be deleted (may already be gone):"
  printf '  %s\n' "${FAILED[@]}"
fi
echo ""
echo "Remaining remote branches:"
git branch -r
