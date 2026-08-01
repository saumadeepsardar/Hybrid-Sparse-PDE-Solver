# =============================================================================
# Makefile  —  Hybrid Adaptive Multilevel Sparse PDE Solver (HSPS)
#              Multi-backend build: OMP / MPI / CUDA
#
# USAGE
#   make                        Build with OMP backend (default)
#   make BACKEND=OMP            Shared-memory OpenMP only
#   make BACKEND=MPI            Hybrid MPI + OpenMP
#   make BACKEND=CUDA           OpenMP + NVIDIA GPU (cuBLAS + cuSPARSE)
#   make BACKEND=MPI_CUDA       All three backends
#
#   make tests                  Build and run all unit tests
#   make examples               Build all example binaries
#   make poisson N=64           Run 2-D Poisson example
#   make convdiff N=64          Run convection-diffusion sweep
#   make bench                  Run energy benchmark
#   make parallel N=64 B=OMP    Run parallel backends example
#   make clean                  Remove all build artefacts
#   make info                   Show detected compiler / flags
#
# EXTRA FLAGS
#   DEBUG=1       -O0 -g  (default: -O3 -march=native)
#   ASAN=1        AddressSanitizer
#   OMP=0         Disable OpenMP even when BACKEND=OMP
#   N=<int>       Grid size for example runners (default 64)
#   B=<backend>   Backend for parallel_backends example (default OMP)
# =============================================================================

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
BACKEND ?= OMP

ifeq ($(BACKEND),MPI)
    USE_MPI  := 1
    USE_OMP  := 1
    USE_CUDA := 0
else ifeq ($(BACKEND),CUDA)
    USE_MPI  := 0
    USE_OMP  := 1
    USE_CUDA := 1
else ifeq ($(BACKEND),MPI_CUDA)
    USE_MPI  := 1
    USE_OMP  := 1
    USE_CUDA := 1
else  # OMP or default
    USE_MPI  := 0
    USE_OMP  := 1
    USE_CUDA := 0
endif

# Allow user to force-disable OMP
ifeq ($(OMP),0)
    USE_OMP := 0
endif

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
ifeq ($(USE_MPI),1)
    CXX := mpic++
else
    CXX := g++
endif

# ── Torch flag (for GNN advisor — Thrust 1) ──────────────────────────────
TORCH ?= 0
ifeq ($(TORCH),1)
    TORCH_HOME    ?= /usr/local/libtorch
    BACKEND_DEFS  += -DHSPS_USE_TORCH
    INCS          += -I$(TORCH_HOME)/include \
                     -I$(TORCH_HOME)/include/torch/csrc/api/include
    LDFLAGS       += -L$(TORCH_HOME)/lib -ltorch -ltorch_cpu -lc10 \
                     -Wl,-rpath,$(TORCH_HOME)/lib
endif

# ── NVML flag (for GPU power monitoring — Thrust 4) ────────────────────────
NVML ?= 0
ifeq ($(NVML),1)
    BACKEND_DEFS  += -DHSPS_USE_NVML
    LDFLAGS       += -lnvidia-ml
endif

ifeq ($(USE_CUDA),1)
    NVCC      := nvcc
    CUDA_HOME ?= /usr/local/cuda
    CUDA_INC  := -I$(CUDA_HOME)/include
    CUDA_LIBS := -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse
else
    NVCC      :=
    CUDA_INC  :=
    CUDA_LIBS :=
endif

AR      := ar
ARFLAGS := rcs

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build
BIN_DIR     := bin
LIB_DIR     := lib
TEST_DIR    := tests
EXAMPLE_DIR := examples

# ---------------------------------------------------------------------------
# Standard and optimisation flags
# ---------------------------------------------------------------------------
STD := -std=c++17

ifeq ($(DEBUG),1)
    OPT_FLAGS := -O0 -g -DDEBUG
else
    OPT_FLAGS := -O3 -march=native -funroll-loops -ffast-math
endif

ifeq ($(ASAN),1)
    OPT_FLAGS := -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
    LDFLAGS   += -fsanitize=address,undefined
endif

# ---------------------------------------------------------------------------
# Warning flags
# ---------------------------------------------------------------------------
WARN_FLAGS := -Wall -Wextra -Wshadow -Wno-unused-parameter \
              -Wno-sign-compare -pedantic

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
ifeq ($(USE_OMP),1)
    OMP_FLAG   := -fopenmp
    OMP_LDFLAG := -fopenmp
else
    OMP_FLAG   :=
    OMP_LDFLAG :=
endif

# ---------------------------------------------------------------------------
# Backend defines
# ---------------------------------------------------------------------------
BACKEND_DEFS :=
ifeq ($(USE_MPI),1)
    BACKEND_DEFS += -DHSPS_USE_MPI
endif
ifeq ($(USE_CUDA),1)
    BACKEND_DEFS += -DHSPS_USE_CUDA
endif

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
INCS := -I$(INC_DIR) $(CUDA_INC)

# ---------------------------------------------------------------------------
# Aggregate CXXFLAGS
# ---------------------------------------------------------------------------
CXXFLAGS := $(STD) $(OPT_FLAGS) $(WARN_FLAGS) $(OMP_FLAG) $(BACKEND_DEFS) $(INCS)

# ---------------------------------------------------------------------------
# Library sources
# ---------------------------------------------------------------------------
LIB_SRCS := \
    $(SRC_DIR)/core/vector.cpp                        \
    $(SRC_DIR)/core/sparse_matrix.cpp                 \
    $(SRC_DIR)/solvers/solver_base.cpp                \
    $(SRC_DIR)/solvers/cg_solver.cpp                  \
    $(SRC_DIR)/solvers/fgmres_solver.cpp              \
    $(SRC_DIR)/solvers/pipelined_fgmres_solver.cpp    \
    $(SRC_DIR)/solvers/solver_factory.cpp             \
    $(SRC_DIR)/solvers/parallel_cg_solver.cpp         \
    $(SRC_DIR)/solvers/parallel_fgmres_solver.cpp     \
    $(SRC_DIR)/preconditioners/jacobi.cpp             \
    $(SRC_DIR)/preconditioners/ilu.cpp                \
    $(SRC_DIR)/preconditioners/amg.cpp                \
    $(SRC_DIR)/energy/energy_monitor.cpp              \
    $(SRC_DIR)/adaptive/adaptive_selector.cpp         \
    $(SRC_DIR)/adaptive/ml_advisor.cpp                \
    $(SRC_DIR)/utils/logger.cpp                       \
    $(SRC_DIR)/utils/matrix_market_io.cpp                       \
    $(SRC_DIR)/parallel/parallel_config.cpp           \
    $(SRC_DIR)/parallel/parallel_context.cpp          \
    $(SRC_DIR)/parallel/backend_base.cpp              \
    $(SRC_DIR)/parallel/omp_backend.cpp               \
    $(SRC_DIR)/parallel/mpi_backend.cpp               \
    $(SRC_DIR)/parallel/cuda_backend.cpp              \
    $(SRC_DIR)/parallel/backend_factory.cpp           \
    $(SRC_DIR)/parallel/distributed_vector.cpp        \
    $(SRC_DIR)/parallel/distributed_matrix.cpp    \
    $(SRC_DIR)/parallel/nvml_monitor.cpp           \
    $(SRC_DIR)/parallel/heterogeneous_context.cpp  \
    $(SRC_DIR)/solvers/sstep_cg_solver.cpp         \
    $(SRC_DIR)/solvers/async_fgmres_solver.cpp     \
    $(SRC_DIR)/energy/roofline_model.cpp           \
    $(SRC_DIR)/adaptive/gnn_advisor.cpp

LIB_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(LIB_SRCS))
LIBRARY   := $(LIB_DIR)/libhsps.a

# ---------------------------------------------------------------------------
# Example binaries
# ---------------------------------------------------------------------------
EXAMPLE_SRCS := \
    $(EXAMPLE_DIR)/poisson_2d.cpp            \
    $(EXAMPLE_DIR)/convection_diffusion.cpp  \
    $(EXAMPLE_DIR)/energy_benchmark.cpp      \
    $(EXAMPLE_DIR)/parallel_backends.cpp \
    $(EXAMPLE_DIR)/run_dataset.cpp

EXAMPLE_BINS := $(patsubst $(EXAMPLE_DIR)/%.cpp, $(BIN_DIR)/%, $(EXAMPLE_SRCS))

# ---------------------------------------------------------------------------
# Test binaries
# ---------------------------------------------------------------------------
TEST_SRCS := \
    $(TEST_DIR)/test_sparse_matrix.cpp    \
    $(TEST_DIR)/test_preconditioners.cpp  \
    $(TEST_DIR)/test_solvers.cpp          \
    $(TEST_DIR)/test_adaptive.cpp         \
    $(TEST_DIR)/test_energy.cpp           \
    $(TEST_DIR)/test_improvements.cpp     \
    $(TEST_DIR)/test_parallel.cpp       \
    $(TEST_DIR)/test_io.cpp

TEST_BINS := $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))

# ---------------------------------------------------------------------------
# Link flags
# ---------------------------------------------------------------------------
LDFLAGS  += $(OMP_LDFLAG) -lm $(CUDA_LIBS)
LDLIBS   := -L$(LIB_DIR) -lhsps

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
.PHONY: all
all: lib examples tests
	@echo ""
	@echo "=== Build complete  BACKEND=$(BACKEND) ==="
	@echo "    Binaries in $(BIN_DIR)/"
	@echo "    MPI=$(USE_MPI)  OMP=$(USE_OMP)  CUDA=$(USE_CUDA)"

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------
.PHONY: lib
lib: $(LIBRARY)

$(LIBRARY): $(LIB_OBJS) | create_build_dirs
	@mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^
	@echo "[AR]  $@  ($(words $(LIB_OBJS)) objects)"

# ---------------------------------------------------------------------------
# Compile rule: .cpp → .o  (with auto-generated dependency files)
# ---------------------------------------------------------------------------
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | create_build_dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
	@echo "[CXX] $<"

-include $(LIB_OBJS:.o=.d)

# ---------------------------------------------------------------------------
# Example binaries
# ---------------------------------------------------------------------------
.PHONY: examples
examples: $(EXAMPLE_BINS)

$(BIN_DIR)/%: $(EXAMPLE_DIR)/%.cpp $(LIBRARY) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) $(LDFLAGS) -o $@
	@echo "[BIN] $@"

# ---------------------------------------------------------------------------
# Test binaries + run
# ---------------------------------------------------------------------------
.PHONY: tests
tests: $(TEST_BINS)
	@echo ""
	@echo "Running unit tests  (BACKEND=$(BACKEND))..."
	@PASS=0; FAIL=0; \
	for t in $(TEST_BINS); do \
	    echo ""; echo ">>> $$t"; \
	    if $$t; then PASS=$$((PASS+1)); else FAIL=$$((FAIL+1)); fi; \
	done; \
	echo ""; \
	echo "========================================"; \
	echo " Test suites: $$((PASS+FAIL))  passed: $$PASS  failed: $$FAIL"; \
	echo " Backend: $(BACKEND)  MPI=$(USE_MPI)  OMP=$(USE_OMP)  CUDA=$(USE_CUDA)"; \
	echo "========================================"; \
	[ $$FAIL -eq 0 ]

$(BIN_DIR)/test_%: $(TEST_DIR)/test_%.cpp $(LIBRARY) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) $(LDFLAGS) -o $@
	@echo "[BIN] $@"

.PHONY: run_tests
run_tests:
	@for t in $(TEST_BINS); do echo ">>> $$t"; $$t; done

# ---------------------------------------------------------------------------
# Example runners
# ---------------------------------------------------------------------------
N ?= 64
B ?= OMP

.PHONY: poisson
poisson: $(BIN_DIR)/poisson_2d
	$(BIN_DIR)/poisson_2d $(N)

.PHONY: convdiff
convdiff: $(BIN_DIR)/convection_diffusion
	$(BIN_DIR)/convection_diffusion $(N)

.PHONY: bench
bench: $(BIN_DIR)/energy_benchmark
	$(BIN_DIR)/energy_benchmark

.PHONY: dataset
dataset: $(BIN_DIR)/run_dataset
	$(BIN_DIR)/run_dataset

.PHONY: dataset_one
dataset_one: $(BIN_DIR)/run_dataset
	$(BIN_DIR)/run_dataset $(MAT)

.PHONY: parallel
parallel: $(BIN_DIR)/parallel_backends
	$(BIN_DIR)/parallel_backends $(B) $(N)

# MPI convenience target
.PHONY: parallel_mpi
parallel_mpi: $(BIN_DIR)/parallel_backends
	mpirun -np $(or $(NP),4) $(BIN_DIR)/parallel_backends MPI $(N) FGMRES

# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------
.PHONY: create_build_dirs
create_build_dirs:
	@mkdir -p $(BUILD_DIR)/core
	@mkdir -p $(BUILD_DIR)/solvers
	@mkdir -p $(BUILD_DIR)/preconditioners
	@mkdir -p $(BUILD_DIR)/energy
	@mkdir -p $(BUILD_DIR)/adaptive
	@mkdir -p $(BUILD_DIR)/utils
	@mkdir -p $(BUILD_DIR)/parallel

$(BIN_DIR):
	@mkdir -p $@

# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------
.PHONY: reproduce_paper1
reproduce_paper1: all
	@echo "=== Reproducing Paper 1 experiments ==="
	@bash scripts/download_suitesparse.sh spd convdiff
	$(BIN_DIR)/energy_benchmark 64
	$(BIN_DIR)/convection_diffusion 64
	$(BIN_DIR)/run_dataset
	@echo "Results in dataset_results.csv"

.PHONY: reproduce_paper2
reproduce_paper2: all
	@echo "=== Reproducing Paper 2 scaling experiments ==="
	@for np in 1 2 4 8; do \
	    echo "--- np=$$np ---"; \
	    $(BIN_DIR)/parallel_backends OMP 64 FGMRES; \
	done

.PHONY: collect_training_data
collect_training_data: $(BIN_DIR)/run_dataset
	@echo "Collecting GNN training data..."
	@bash scripts/collect_dataset.sh
	@echo "Run: python3 scripts/train_gnn.py --input data/training.jsonl"

.PHONY: train_gnn
train_gnn:
	python3 scripts/train_gnn.py \
	    --input  data/training.jsonl \
	    --output models/gnn_v1.pt   \
	    --epochs 200

.PHONY: calibrate
calibrate:
	python3 scripts/calibrate_energy.py --output models/energy_coeffs.json

.PHONY: download_matrices
download_matrices:
	bash scripts/download_suitesparse.sh $(or $(CAT),all)

.PHONY: info
info:
	@echo "BACKEND   = $(BACKEND)"
	@echo "CXX       = $(CXX)"
	@echo "CXXFLAGS  = $(CXXFLAGS)"
	@echo "LDFLAGS   = $(LDFLAGS)"
	@echo "USE_MPI   = $(USE_MPI)"
	@echo "USE_OMP   = $(USE_OMP)"
	@echo "USE_CUDA  = $(USE_CUDA)"
	@echo "CUDA_HOME = $(CUDA_HOME)"
	@echo "TORCH     = $(TORCH)  TORCH_HOME=$(TORCH_HOME)"
	@echo "NVML      = $(NVML)"
	@$(CXX) --version | head -1

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR)
	@rm -f *.csv /tmp/hsps_*.csv /tmp/hsps_*.csv
	@echo "Clean complete."

.SECONDARY: $(LIB_OBJS)
