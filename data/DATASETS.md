# HSPS Dataset Guide — Real PDE Sparse Matrices

## Matrices in this `data/` directory (ready to use now)

| File | n | nnz | Type | Origin |
|------|---|-----|------|--------|
| `pde225.mtx`              | 225   | 1,065  | 2D Laplacian 15×15 | Mirrors Bai/pde225 (SuiteSparse) |
| `pde900.mtx`              | 900   | 4,380  | 2D Laplacian 30×30 | Mirrors Bai/pde900 (SuiteSparse) |
| `pde2961.mtx`             | 2,916 | 14,220 | 2D Laplacian 54×54 | Mirrors Bai/pde2961 structure |
| `helmholtz_400.mtx`       | 400   | 1,960  | Helmholtz (indefinite) | Custom, k=2.5 |
| `convdiff_upwind_484.mtx` | 484   | 2,332  | Conv-Diff asymmetric | Custom, ε=0.05 |

Each has a matching `*_rhs.mtx` file. The runner uses them automatically.

---

## Run commands

```bash
# Run all 5 bundled matrices
./bin/run_dataset

# Run a single matrix
./bin/run_dataset data/pde900.mtx

# Run with explicit RHS
./bin/run_dataset data/pde900.mtx data/pde900_rhs.mtx

# Run a whole directory (e.g., after downloading more)
./bin/run_dataset /path/to/my_matrices/
```

---

## Downloading real matrices from SuiteSparse (sparse.tamu.edu)

The SuiteSparse Matrix Collection hosts 3,000+ real-application PDE matrices.
Download format: MatrixMarket (.mtx) — exactly what our reader expects.

### Download script (run from project root)

```bash
cd data/

# Function: download and unpack one matrix
get_matrix() {
    GROUP=$1; NAME=$2
    URL="https://suitesparse-collection-website.herokuapp.com/MM/${GROUP}/${NAME}.tar.gz"
    wget -q "${URL}" -O "${NAME}.tar.gz" && tar -xzf "${NAME}.tar.gz" \
        && mv "${NAME}/${NAME}.mtx" . && rm -rf "${NAME}" "${NAME}.tar.gz"
    echo "Downloaded: ${NAME}.mtx"
}

# ── Elliptic PDEs (SPD, good for CG+AMG) ────────────────────────────────────
get_matrix Bai     pde225        # 2D Laplacian 15×15,  n=225
get_matrix Bai     pde900        # 2D Laplacian 30×30,  n=900
get_matrix Bai     pde2961       # 2D Laplacian,        n=2,961
get_matrix FEMLAB  poisson2D     # FEM Poisson,         n=74,752
get_matrix 2cubes_sphere         # 3D Laplacian sphere, n=101,492

# ── 3D Structural / Elasticity (SPD, large-scale AMG target) ────────────────
get_matrix Boeing   bcsstk38    # Structural, n=8,032,  nnz=355k   SPD
get_matrix Boeing   bcsstk13    # Structural, n=2,003,  nnz=42k    SPD
get_matrix AMD      G3_circuit  # 3D circuit, n=1.5M,  nnz=7.7M  SPD

# ── Convection-Diffusion (non-symmetric, tests FGMRES+ILU/AMG) ──────────────
get_matrix Bai     olm500       # Conv-Diff, n=500,   non-sym
get_matrix Bai     olm5000      # Conv-Diff, n=5000,  non-sym
get_matrix GHS_indef  helm2d03  # Helmholtz 2D, n=392,257  indefinite

# ── Fluid Dynamics (stiff, non-symmetric) ───────────────────────────────────
get_matrix Cylshell  s3rmt3m3   # CFD shell, n=5,357,  nnz=207k
get_matrix DNVS      ship_003   # Ship structure, n=121k

# ── Heat / Thermal (SPD, medium scale) ──────────────────────────────────────
get_matrix Norris    heart1     # Cardiac FEM, n=3,557
get_matrix Oberwolfach  bfly    # Butterfly PDE, n=10,467
```

### Verify downloads
```bash
ls -lh data/*.mtx | awk '{print $5, $9}'
```

---

## Which matrix → which solver? (Expected adaptive routing)

| Matrix class | Condition | Expected state | Why |
|---|---|---|---|
| 2D/3D Laplacian (fine grid) | κ ≈ O(n²/π²) | **EASY** → CG+Jacobi | SPD, moderate condition |
| Large 3D FEM (>10k DOFs) | κ >> 10⁴ | **HARD** → FGMRES+AMG | Ill-conditioned, benefits from AMG hierarchy |
| Convection-dominated ε<0.1 | non-symmetric | **MODERATE/HARD** → FGMRES+ILU | Non-symmetric, ILU better |
| Helmholtz (k²>0) | indefinite | **HARD** → FGMRES+AMG | Indefinite, needs flexible precond |
| Structural bcsstk* | SPD, high cond | **HARD** → FGMRES+AMG | High condition number |

---

## Reading the output columns

```
Matrix                       n      nnz   SPD     Final State          Iters  Esc   Rel-Res    Time(s)   Energy(J)  Conv
pde225                     225    1065    yes  EASY  [CG+Jacobi]          43    0  2.34e-09    0.0002   4.12e-08   YES
pde900                     900    4380    yes  EASY  [CG+Jacobi]         162    0  5.67e-09    0.0014   1.73e-07   YES
helmholtz_400              400    1960    no   HARD  [FGMRES+AMG]         28    2  3.21e-09    0.0021   2.89e-07   YES
convdiff_upwind_484        484    2332    no   MODERATE [FGMRES+ILU]      61    1  8.14e-09    0.0019   2.11e-07   YES
```

- **Esc** = number of adaptive escalations (0 = solved at initial state)
- **Final State** = which solver/preconditioner actually converged it
- The results CSV (`dataset_results.csv`) has all fields for plotting

---

## Using your own PDE data

Any `.mtx` file in MatrixMarket coordinate real format works:

```
%%MatrixMarket matrix coordinate real symmetric
% My custom PDE matrix
1000 1000 4950
1 1 4.0
1 2 -1.0
2 1 -1.0
...
```

Then:
```bash
cp my_matrix.mtx data/
./bin/run_dataset data/my_matrix.mtx
```

The loader handles both `symmetric` (upper-triangle only) and `general`
(all entries) formats automatically.

---

## References

- SuiteSparse Matrix Collection: https://sparse.tamu.edu
- Davis & Hu (2011): "The University of Florida Sparse Matrix Collection",
  ACM TOMS 38(1), DOI: 10.1145/2049662.2049663
- MatrixMarket format spec: https://math.nist.gov/MatrixMarket/formats.html
