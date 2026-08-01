// =============================================================================
// matrix_market_io.cpp  —  MatrixMarket (.mtx) reader / writer
// =============================================================================

#include "../../include/utils/matrix_market_io.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <map>
#include <cmath>

#ifdef __has_include
#  if __has_include(<filesystem>)
#    include <filesystem>
#    define HSPS_HAS_FILESYSTEM 1
     namespace fs = std::filesystem;
#  endif
#endif

namespace hsps {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

static std::ifstream open_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.good())
        throw std::runtime_error("MatrixMarketIO: cannot open '" + path + "'");
    return f;
}

// ---------------------------------------------------------------------------
// parse_header_line  —  the %%MatrixMarket banner
// ---------------------------------------------------------------------------
MatrixMarketHeader MatrixMarketIO::parse_header_line(const std::string& line) {
    MatrixMarketHeader h;
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token) tokens.push_back(to_lower(token));

    // tokens: %%matrixmarket  matrix  coordinate|array  real|integer|complex|pattern  general|symmetric|...
    if (tokens.size() < 5)
        throw std::runtime_error("MatrixMarketIO: invalid header: " + line);

    // format
    if (tokens[2] == "coordinate") h.format = MatrixMarketHeader::Format::COORDINATE;
    else if (tokens[2] == "array") h.format = MatrixMarketHeader::Format::ARRAY;
    else throw std::runtime_error("MatrixMarketIO: unknown format: " + tokens[2]);

    // field
    if      (tokens[3] == "real")    h.field = MatrixMarketHeader::Field::REAL;
    else if (tokens[3] == "integer") h.field = MatrixMarketHeader::Field::INTEGER;
    else if (tokens[3] == "complex") h.field = MatrixMarketHeader::Field::COMPLEX;
    else if (tokens[3] == "pattern") h.field = MatrixMarketHeader::Field::PATTERN;
    else throw std::runtime_error("MatrixMarketIO: unknown field: " + tokens[3]);

    // symmetry
    if      (tokens[4] == "general")          h.symmetry = MatrixMarketHeader::Symmetry::GENERAL;
    else if (tokens[4] == "symmetric")        h.symmetry = MatrixMarketHeader::Symmetry::SYMMETRIC;
    else if (tokens[4] == "skew-symmetric")   h.symmetry = MatrixMarketHeader::Symmetry::SKEW_SYMMETRIC;
    else if (tokens[4] == "hermitian")        h.symmetry = MatrixMarketHeader::Symmetry::HERMITIAN;
    else throw std::runtime_error("MatrixMarketIO: unknown symmetry: " + tokens[4]);

    return h;
}

// ---------------------------------------------------------------------------
// read_header  —  parse header and size line only
// ---------------------------------------------------------------------------
MatrixMarketHeader MatrixMarketIO::read_header(const std::string& path) {
    auto f = open_file(path);
    std::string line;

    // First line must be the banner
    std::getline(f, line);
    if (line.size() < 2 || line[0] != '%' || line[1] != '%')
        throw std::runtime_error("MatrixMarketIO: missing %% banner in " + path);

    MatrixMarketHeader h = parse_header_line(line);

    // Skip comment lines (lines starting with %)
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        // This is the size line
        std::istringstream iss(line);
        if (h.format == MatrixMarketHeader::Format::COORDINATE) {
            iss >> h.rows >> h.cols >> h.nnz;
            h.nnz_full = (h.symmetry == MatrixMarketHeader::Symmetry::GENERAL)
                         ? h.nnz
                         : h.nnz * 2 - h.rows;  // diagonal entries not doubled
        } else { // ARRAY
            iss >> h.rows >> h.cols;
            h.nnz = h.rows * h.cols;
            h.nnz_full = h.nnz;
        }
        break;
    }
    return h;
}

// ---------------------------------------------------------------------------
// load_matrix
// ---------------------------------------------------------------------------
SparseMatrix MatrixMarketIO::load_matrix(const std::string& path) {
    auto f = open_file(path);
    std::string line;

    // Banner
    std::getline(f, line);
    if (line.size() < 2 || line[0] != '%' || line[1] != '%')
        throw std::runtime_error("MatrixMarketIO: missing %% banner in " + path);
    MatrixMarketHeader h = parse_header_line(line);

    if (h.format != MatrixMarketHeader::Format::COORDINATE)
        throw std::runtime_error(
            "MatrixMarketIO: load_matrix requires coordinate format, got array in " + path);

    if (h.field == MatrixMarketHeader::Field::COMPLEX)
        throw std::runtime_error(
            "MatrixMarketIO: complex matrices not supported (use real part only)");

    // Skip comments, read size line
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss(line);
        iss >> h.rows >> h.cols >> h.nnz;
        break;
    }

    // Estimate full nnz for reservation
    size_t reserve_n = (h.symmetry == MatrixMarketHeader::Symmetry::GENERAL)
                       ? h.nnz
                       : h.nnz * 2;

    std::vector<Index> row_vec, col_vec;
    std::vector<Real>  val_vec;
    row_vec.reserve(reserve_n);
    col_vec.reserve(reserve_n);
    val_vec.reserve(reserve_n);

    int entries_read = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss(line);
        int r, c; double v = 1.0;
        iss >> r >> c;
        if (h.field != MatrixMarketHeader::Field::PATTERN) iss >> v;

        // Convert 1-indexed → 0-indexed
        --r; --c;
        row_vec.push_back(static_cast<Index>(r));
        col_vec.push_back(static_cast<Index>(c));
        val_vec.push_back(static_cast<Real>(v));

        // Expand symmetry: off-diagonal entry (r,c) → also add (c,r)
        if (h.symmetry == MatrixMarketHeader::Symmetry::SYMMETRIC && r != c) {
            row_vec.push_back(static_cast<Index>(c));
            col_vec.push_back(static_cast<Index>(r));
            val_vec.push_back(static_cast<Real>(v));
        }
        if (h.symmetry == MatrixMarketHeader::Symmetry::SKEW_SYMMETRIC && r != c) {
            row_vec.push_back(static_cast<Index>(c));
            col_vec.push_back(static_cast<Index>(r));
            val_vec.push_back(static_cast<Real>(-v));
        }
        ++entries_read;
        if (entries_read == h.nnz) break;
    }

    return SparseMatrix::from_coo(
        static_cast<Index>(h.rows),
        static_cast<Index>(h.cols),
        row_vec, col_vec, val_vec);
}

// ---------------------------------------------------------------------------
// load_vector
// ---------------------------------------------------------------------------
Vector MatrixMarketIO::load_vector(const std::string& path, int expected_n) {
    auto f = open_file(path);
    std::string line;

    // Banner
    std::getline(f, line);
    MatrixMarketHeader h;
    if (line.size() >= 2 && line[0] == '%' && line[1] == '%') {
        h = parse_header_line(line);
    }
    // else: plain text list — no banner

    int n_entries = 0;
    bool found_size = false;

    if (h.format == MatrixMarketHeader::Format::ARRAY ||
        h.format == MatrixMarketHeader::Format::COORDINATE) {
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '%') continue;
            std::istringstream iss(line);
            if (h.format == MatrixMarketHeader::Format::ARRAY) {
                int rows, cols;
                iss >> rows >> cols;
                n_entries = rows * cols;
            } else {
                // Could be "N 1 N" or "N"
                int a, b, c_val;
                iss >> a;
                if (iss >> b) { if (iss >> c_val) n_entries = c_val; else n_entries = a; }
                else n_entries = a;
            }
            found_size = true;
            break;
        }
    }

    if (expected_n > 0 && !found_size) n_entries = expected_n;

    Vector result;
    result.resize(n_entries > 0 ? n_entries : (expected_n > 0 ? expected_n : 0));

    int idx = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss(line);
        double v;
        // Coordinate vector: might have "row col val" or just "val"
        if (h.format == MatrixMarketHeader::Format::COORDINATE) {
            int r; double val;
            if (iss >> r >> val) {
                if (r - 1 < result.size()) result[r - 1] = static_cast<Real>(val);
            }
        } else {
            if (iss >> v && idx < result.size())
                result[idx++] = static_cast<Real>(v);
        }
    }

    // Pad to expected size if needed
    if (expected_n > 0 && result.size() != static_cast<size_t>(expected_n))
        result.resize(expected_n, REAL_ONE);

    return result;
}

// ---------------------------------------------------------------------------
// load_problem
// ---------------------------------------------------------------------------
std::pair<SparseMatrix, Vector>
MatrixMarketIO::load_problem(const std::string& matrix_path,
                              const std::string& rhs_path) {
    SparseMatrix A = load_matrix(matrix_path);
    Vector b;

    if (!rhs_path.empty()) {
        // Try to load RHS; fall back to default on failure
        try {
            b = load_vector(rhs_path, A.rows());
            if (b.size() != static_cast<size_t>(A.rows()))
                b = make_default_rhs(A.rows());
        } catch (...) {
            std::cerr << "[MatrixMarketIO] Warning: could not load RHS from '"
                      << rhs_path << "', using default b=1\n";
            b = make_default_rhs(A.rows());
        }
    } else {
        b = make_default_rhs(A.rows());
    }
    return {std::move(A), std::move(b)};
}

// ---------------------------------------------------------------------------
// make_default_rhs
// ---------------------------------------------------------------------------
Vector MatrixMarketIO::make_default_rhs(int n, Real value) {
    return Vector(n, value);
}

// ---------------------------------------------------------------------------
// save_vector
// ---------------------------------------------------------------------------
void MatrixMarketIO::save_vector(const std::string& path,
                                  const Vector& v,
                                  const std::string& comment) {
    std::ofstream f(path);
    if (!f.good())
        throw std::runtime_error("MatrixMarketIO: cannot write to '" + path + "'");

    f << "%%MatrixMarket matrix array real general\n";
    if (!comment.empty()) f << "% " << comment << "\n";
    f << v.size() << " 1\n";
    f << std::scientific << std::setprecision(15);
    for (Index i = 0; i < v.size(); ++i)
        f << v[i] << "\n";
}

// ---------------------------------------------------------------------------
// save_matrix
// ---------------------------------------------------------------------------
void MatrixMarketIO::save_matrix(const std::string& path,
                                  const SparseMatrix& A,
                                  const std::string& comment) {
    std::ofstream f(path);
    if (!f.good())
        throw std::runtime_error("MatrixMarketIO: cannot write to '" + path + "'");

    f << "%%MatrixMarket matrix coordinate real general\n";
    if (!comment.empty()) f << "% " << comment << "\n";
    f << A.rows() << " " << A.cols() << " " << A.nnz() << "\n";
    f << std::scientific << std::setprecision(15);

    const auto& rp  = A.row_ptr();
    const auto& ci  = A.col_idx();
    const auto& val = A.values();
    for (Index i = 0; i < A.rows(); ++i)
        for (Index k = rp[i]; k < rp[i + 1]; ++k)
            f << (i + 1) << " " << (ci[k] + 1) << " " << val[k] << "\n";
}

// ---------------------------------------------------------------------------
// list_mtx_files
// ---------------------------------------------------------------------------
std::vector<std::string> MatrixMarketIO::list_mtx_files(const std::string& dir) {
    std::vector<std::string> result;
#ifdef HSPS_HAS_FILESYSTEM
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".mtx") {
                // Skip RHS files (they end in _rhs.mtx or _b.mtx)
                std::string stem = entry.path().stem().string();
                if (stem.size() > 4 && stem.substr(stem.size() - 4) == "_rhs") continue;
                if (stem.size() > 2 && stem.substr(stem.size() - 2) == "_b") continue;
                result.push_back(entry.path().string());
            }
        }
        std::sort(result.begin(), result.end());
    } catch (...) {}
#else
    // Fallback: return known data directory files
    (void)dir;
#endif
    return result;
}

// ---------------------------------------------------------------------------
// kind_string
// ---------------------------------------------------------------------------
std::string MatrixMarketHeader::kind_string() const {
    std::string s;
    switch (format) {
        case Format::COORDINATE: s += "coordinate "; break;
        case Format::ARRAY:      s += "array ";      break;
    }
    switch (field) {
        case Field::REAL:    s += "real ";    break;
        case Field::INTEGER: s += "integer "; break;
        case Field::COMPLEX: s += "complex "; break;
        case Field::PATTERN: s += "pattern "; break;
    }
    switch (symmetry) {
        case Symmetry::GENERAL:         s += "general";         break;
        case Symmetry::SYMMETRIC:       s += "symmetric";       break;
        case Symmetry::SKEW_SYMMETRIC:  s += "skew-symmetric";  break;
        case Symmetry::HERMITIAN:       s += "hermitian";       break;
    }
    return s;
}

} // namespace hsps
