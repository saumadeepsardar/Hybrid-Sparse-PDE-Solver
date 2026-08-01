#pragma once

// =============================================================================
// matrix_market_io.hpp  —  Read/Write MatrixMarket (.mtx) files
//
// Supports
// --------
//   %%MatrixMarket matrix coordinate real general
//   %%MatrixMarket matrix coordinate real symmetric   (expands to full)
//   %%MatrixMarket matrix coordinate real skew-symmetric
//   %%MatrixMarket matrix array real general           (dense vector/matrix)
//
// Usage: loading a matrix + optional RHS
// ----------------------------------------
//   SparseMatrix A;
//   Vector       b;
//   MatrixMarketIO::load_matrix("data/pde900.mtx",     A);
//   MatrixMarketIO::load_vector("data/pde900_rhs.mtx", b, A.rows());
//
//   // If no RHS file is provided, a default b=1 vector is created:
//   if (b.empty()) b = MatrixMarketIO::make_default_rhs(A.rows());
//
// Usage: write a solution or residual
// -------------------------------------
//   MatrixMarketIO::save_vector("results/x.mtx", x, "solution");
//
// SuiteSparse download URLs
// --------------------------
//   The format this reader accepts is exactly what sparse.tamu.edu
//   distributes.  To download any matrix:
//
//   wget "https://suitesparse-collection-website.herokuapp.com/MM/<Group>/<name>.tar.gz"
//   tar -xzf <name>.tar.gz
//   # → <name>/<name>.mtx  (and optionally <name>/<name>_b.mtx for RHS)
//
// Recommended PDE matrices from SuiteSparse
// -------------------------------------------
// See data/DATASETS.md for the full curated list.
// =============================================================================

#include "../core/types.hpp"
#include "../core/sparse_matrix.hpp"
#include "../core/vector.hpp"

#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <iosfwd>

namespace hsps {

// ---------------------------------------------------------------------------
// Matrix metadata extracted from the header
// ---------------------------------------------------------------------------
struct MatrixMarketHeader {
    enum class Format   { COORDINATE, ARRAY };
    enum class Field    { REAL, INTEGER, COMPLEX, PATTERN };
    enum class Symmetry { GENERAL, SYMMETRIC, SKEW_SYMMETRIC, HERMITIAN };

    Format   format   = Format::COORDINATE;
    Field    field    = Field::REAL;
    Symmetry symmetry = Symmetry::GENERAL;

    int rows = 0;
    int cols = 0;
    int nnz  = 0;      ///< Entries in file (before symmetry expansion)
    int nnz_full = 0;  ///< Entries after expansion

    bool is_square()    const { return rows == cols; }
    bool is_symmetric() const { return symmetry == Symmetry::SYMMETRIC; }
    bool is_spd_candidate() const { return is_square() && is_symmetric(); }

    std::string kind_string() const;
};

// ---------------------------------------------------------------------------
// MatrixMarketIO
// ---------------------------------------------------------------------------
class MatrixMarketIO {
public:
    // ------------------------------------------------------------------
    // Load matrix from .mtx file
    // ------------------------------------------------------------------

    /// Load a sparse matrix, expanding symmetric storage to full format.
    static SparseMatrix load_matrix(const std::string& path);

    /// Load a dense vector from an array-format .mtx file.
    /// If expected_n > 0, pads or truncates to that size.
    static Vector load_vector(const std::string& path, int expected_n = 0);

    /// Load matrix header only (no data) — fast metadata query.
    static MatrixMarketHeader read_header(const std::string& path);

    // ------------------------------------------------------------------
    // Convenience: load matrix + RHS together
    //   rhs_path = ""  → creates unit RHS of length A.rows()
    // ------------------------------------------------------------------
    static std::pair<SparseMatrix, Vector>
    load_problem(const std::string& matrix_path,
                 const std::string& rhs_path = "");

    // ------------------------------------------------------------------
    // Make a sensible default RHS when none is provided
    // ------------------------------------------------------------------
    static Vector make_default_rhs(int n, Real value = 1.0);

    // ------------------------------------------------------------------
    // Save a vector to .mtx (array format)
    // ------------------------------------------------------------------
    static void save_vector(const std::string& path,
                            const Vector& v,
                            const std::string& comment = "");

    // ------------------------------------------------------------------
    // Save a sparse matrix to .mtx (coordinate real general)
    // ------------------------------------------------------------------
    static void save_matrix(const std::string& path,
                            const SparseMatrix& A,
                            const std::string& comment = "");

    // ------------------------------------------------------------------
    // List .mtx files in a directory
    // ------------------------------------------------------------------
    static std::vector<std::string> list_mtx_files(const std::string& dir);

private:
    static MatrixMarketHeader parse_header_line(const std::string& line);
};

} // namespace hsps
