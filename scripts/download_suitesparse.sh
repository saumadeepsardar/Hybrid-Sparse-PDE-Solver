#!/bin/bash
# =============================================================================
# scripts/download_suitesparse.sh  —  Download curated PDE matrices
#
# Downloads 15+ real-application matrices from the SuiteSparse Matrix
# Collection (sparse.tamu.edu) into the data/ directory.
# All matrices are in MatrixMarket format — directly readable by
# MatrixMarketIO::load_matrix().
#
# Usage:
#   bash scripts/download_suitesparse.sh              # download all
#   bash scripts/download_suitesparse.sh spd          # SPD matrices only
#   bash scripts/download_suitesparse.sh convdiff     # conv-diff only
#   bash scripts/download_suitesparse.sh helmholtz    # Helmholtz only
#
# After downloading, run:
#   ./bin/run_dataset                   # solve all in data/
#   bash scripts/collect_dataset.sh     # collect GNN training data
# =============================================================================

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"
WGET_OPTS="--timeout=60 --tries=3 -q --show-progress"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

DOWNLOADED=0
FAILED=0

get_matrix() {
    local GROUP="$1"
    local NAME="$2"
    local LABEL="${3:-}"

    if [[ -f "${NAME}.mtx" ]]; then
        echo "  [skip]  ${NAME}.mtx (already exists)"
        return 0
    fi

    local URL="${BASE_URL}/${GROUP}/${NAME}.tar.gz"
    echo -n "  Downloading ${NAME} (${LABEL})... "

    if wget $WGET_OPTS "${URL}" -O "${NAME}.tar.gz" 2>/dev/null; then
        if tar -xzf "${NAME}.tar.gz" --strip-components=1 "${NAME}/${NAME}.mtx" 2>/dev/null \
           || tar -xzf "${NAME}.tar.gz" 2>/dev/null; then
            # Move to data root if in subdirectory
            [[ -d "${NAME}" ]] && mv "${NAME}/${NAME}.mtx" . && rm -rf "${NAME}"
            rm -f "${NAME}.tar.gz"
            local SIZE
            SIZE=$(ls -lh "${NAME}.mtx" 2>/dev/null | awk '{print $5}')
            echo "OK (${SIZE})"
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            echo "UNPACK FAILED"
            rm -f "${NAME}.tar.gz"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "DOWNLOAD FAILED (check network / URL)"
        rm -f "${NAME}.tar.gz"
        FAILED=$((FAILED + 1))
    fi
}

CATEGORY="${1:-all}"

# ── SPD Elliptic PDEs (good for CG+AMG, Papers 1 & 2) ─────────────────────
if [[ "$CATEGORY" == "all" || "$CATEGORY" == "spd" ]]; then
    echo ""
    echo "=== SPD Elliptic PDEs ==="
    get_matrix "Bai"     "pde225"       "2D Laplacian n=225,   SPD"
    get_matrix "Bai"     "pde900"       "2D Laplacian n=900,   SPD"
    get_matrix "Bai"     "pde2961"      "2D Laplacian n=2961,  SPD"
    get_matrix "FEMLAB"  "poisson2D"    "FEM Poisson  n=74752, SPD"
    get_matrix "Norris"  "heart1"       "Cardiac FEM  n=3557,  SPD"
fi

# ── 3D Structural / Elasticity (large-scale AMG target, Paper 2) ───────────
if [[ "$CATEGORY" == "all" || "$CATEGORY" == "structural" ]]; then
    echo ""
    echo "=== 3D Structural ==="
    get_matrix "Boeing"  "bcsstk13"     "Structural   n=2003,  SPD"
    get_matrix "Boeing"  "bcsstk38"     "Structural   n=8032,  SPD"
    get_matrix "AMD"     "G3_circuit"   "3D circuit   n=1.5M,  SPD — LARGE"
fi

# ── Convection-Diffusion (non-symmetric, FGMRES+ILU target) ────────────────
if [[ "$CATEGORY" == "all" || "$CATEGORY" == "convdiff" ]]; then
    echo ""
    echo "=== Convection-Diffusion (non-symmetric) ==="
    get_matrix "Bai"     "olm500"       "Conv-Diff    n=500,   non-sym"
    get_matrix "Bai"     "olm5000"      "Conv-Diff    n=5000,  non-sym"
    get_matrix "Bai"     "rdb450"       "Conv-Diff    n=450,   non-sym"
    get_matrix "Bai"     "rdb968"       "Conv-Diff    n=968,   non-sym"
fi

# ── Helmholtz / Indefinite (FGMRES+AMG, Paper 1 stress test) ───────────────
if [[ "$CATEGORY" == "all" || "$CATEGORY" == "helmholtz" ]]; then
    echo ""
    echo "=== Helmholtz / Indefinite ==="
    get_matrix "GHS_indef" "helm2d03"   "Helmholtz 2D n=392257, indefinite — LARGE"
    get_matrix "Bai"       "tols340"    "Helmholtz    n=340,  non-SPD"
    get_matrix "Bai"       "tols90"     "Helmholtz    n=90,   non-SPD"
fi

# ── Thermal / Heat (SPD, medium scale) ─────────────────────────────────────
if [[ "$CATEGORY" == "all" || "$CATEGORY" == "thermal" ]]; then
    echo ""
    echo "=== Thermal / Heat ==="
    get_matrix "Oberwolfach" "bfly"     "Butterfly PDE n=10467, SPD"
    get_matrix "Norris"      "lung2"    "Lung FEM      n=109460, SPD"
fi

cd - > /dev/null

echo ""
echo "=========================================="
echo "  Downloaded: ${DOWNLOADED}  Failed: ${FAILED}"
echo "  Matrices in ${DATA_DIR}/:"
ls -1 "${DATA_DIR}"/*.mtx 2>/dev/null | grep -v "_rhs\|_b\.mtx\|_sol" \
    | while read f; do
        name=$(basename "$f" .mtx)
        size=$(ls -lh "$f" | awk '{print $5}')
        echo "    ${name}  (${size})"
    done
echo "=========================================="
echo ""
echo "Run:  ./bin/run_dataset"
echo "Then: bash scripts/collect_dataset.sh"
