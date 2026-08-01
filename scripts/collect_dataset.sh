#!/bin/bash
# =============================================================================
# scripts/collect_dataset.sh  —  Collect GNN training data (Thrust 1)
#
# Runs bin/run_dataset on every matrix in data/ cycling through all
# (solver_state, restart_size, drop_tol) combinations and writes a unified
# data/training.jsonl file for train_gnn.py.
#
# Usage:
#   bash scripts/collect_dataset.sh                  # all matrices in data/
#   bash scripts/collect_dataset.sh data/pde900.mtx  # single matrix
#   REPEATS=10 bash scripts/collect_dataset.sh        # more samples per config
#
# Time estimate:
#   5 matrices × 3 states × 3 restarts × 5 repeats = 225 solve calls
#   On a laptop: ~45 minutes
#   On a workstation with 16 cores: ~8 minutes
#
# Output:
#   data/training.jsonl         (all triples, JSON Lines)
#   data/training_summary.txt   (statistics)
# =============================================================================

set -euo pipefail

BINARY="./bin/run_dataset"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT="${OUTPUT:-data/training.jsonl}"
REPEATS="${REPEATS:-5}"

# Solver states: 0=EASY (CG+Jacobi), 1=MODERATE (FGMRES+ILU), 2=HARD (FGMRES+AMG)
STATES=(0 1 2)
STATE_NAMES=("EASY" "MODERATE" "HARD")

# Restart sizes to sweep (for FGMRES)
RESTARTS=(20 50 100)

# Drop tolerances for ILUT (affects MODERATE state quality)
# Passed as --restart override; drop_tol tuning requires a separate flag (future)
# For now we vary restart to get diverse (features, config) coverage

# Check binary exists
if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: $BINARY not found. Run 'make' first."
    exit 1
fi

# Collect matrix files
if [[ $# -ge 1 ]] && [[ -f "$1" ]]; then
    MATRICES=("$1")
else
    mapfile -t MATRICES < <(find "$DATA_DIR" -name "*.mtx" \
        ! -name "*_rhs*" ! -name "*_b.mtx" ! -name "*_sol*" | sort)
fi

echo "=========================================="
echo "  HAMLSS Dataset Collection"
echo "  Matrices: ${#MATRICES[@]}"
echo "  States:   ${#STATES[@]}"
echo "  Restarts: ${#RESTARTS[@]}"
echo "  Repeats:  ${REPEATS}"
echo "  Output:   ${OUTPUT}"
echo "  Total runs: $((${#MATRICES[@]} * ${#STATES[@]} * ${#RESTARTS[@]} * REPEATS))"
echo "=========================================="

# Clear existing output if it exists
if [[ -f "$OUTPUT" ]]; then
    echo "Appending to existing ${OUTPUT} ..."
else
    touch "$OUTPUT"
    echo "Created ${OUTPUT}"
fi

TOTAL_RECORDS=0
FAILED=0
START_TIME=$(date +%s)

for MAT in "${MATRICES[@]}"; do
    MAT_NAME=$(basename "$MAT" .mtx)
    echo ""
    echo "--- Matrix: ${MAT_NAME} ---"

    for STATE_IDX in "${!STATES[@]}"; do
        STATE="${STATES[$STATE_IDX]}"
        STATE_NAME="${STATE_NAMES[$STATE_IDX]}"

        for RESTART in "${RESTARTS[@]}"; do
            echo -n "  ${STATE_NAME}  restart=${RESTART}  ×${REPEATS} ... "

            if "$BINARY" "$MAT" \
                    --force-state "$STATE_NAME" \
                    --restart "$RESTART" \
                    --repeat "$REPEATS" \
                    --output-jsonl "$OUTPUT" \
                    2>/dev/null; then
                TOTAL_RECORDS=$((TOTAL_RECORDS + REPEATS))
                echo "OK"
            else
                FAILED=$((FAILED + 1))
                echo "FAILED (skipped)"
            fi
        done
    done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "  Collection complete"
echo "  Total records written : ${TOTAL_RECORDS}"
echo "  Failed configurations : ${FAILED}"
echo "  Elapsed time          : ${ELAPSED}s"
echo "  Output file           : ${OUTPUT}"

# ── Summary statistics ──────────────────────────────────────────────────────
SUMMARY="${DATA_DIR}/training_summary.txt"
{
    echo "HAMLSS Dataset Summary"
    echo "Generated: $(date)"
    echo ""
    echo "Matrices: ${#MATRICES[@]}"
    for m in "${MATRICES[@]}"; do echo "  $(basename $m)"; done
    echo ""
    echo "States × Restarts × Repeats: ${#STATES[@]} × ${#RESTARTS[@]} × ${REPEATS}"
    echo "Total records: ${TOTAL_RECORDS}"
    echo "Failed: ${FAILED}"
    echo ""
    echo "Next steps:"
    echo "  python3 scripts/train_gnn.py --input ${OUTPUT} --output models/gnn_v1.pt"
} > "$SUMMARY"

echo "  Summary                : ${SUMMARY}"
echo "=========================================="
echo ""
echo "Next step:"
echo "  python3 scripts/train_gnn.py --input ${OUTPUT} --output models/gnn_v1.pt --epochs 200"
