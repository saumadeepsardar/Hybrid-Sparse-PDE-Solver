#!/usr/bin/env python3
"""
scripts/calibrate_energy.py  —  Measure platform energy coefficients (Thrust 2)

Runs STREAM Triad and dot-product micro-benchmarks, reads RAPL or estimates
from timing, and writes energy_coeffs.json for the C++ RooflineModel.

Usage:
    python3 scripts/calibrate_energy.py --output models/energy_coeffs.json
    python3 scripts/calibrate_energy.py --platform server --repeat 5

Requirements:
    pip install numpy
    Intel RAPL: run as root OR add user to /sys/class/powercap perms
    AMD uProf:  amduprof-cli (optional, auto-detected)
    NVML:       pynvml  (pip install pynvml, optional)
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Optional hardware interfaces ───────────────────────────────────────────
RAPL_AVAILABLE  = False
NVML_AVAILABLE  = False
AMUPROF_AVAILABLE = False

def try_read_rapl():
    """Read RAPL package-0 energy in joules. Returns None if unavailable."""
    paths = [
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
    ]
    for p in paths:
        try:
            with open(p) as f:
                return float(f.read().strip()) * 1e-6  # μJ → J
        except (OSError, PermissionError):
            pass
    return None

RAPL_AVAILABLE = try_read_rapl() is not None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    pass

# ── Benchmark kernels ──────────────────────────────────────────────────────

def stream_triad(n: int, repeat: int = 3):
    """STREAM Triad: C[i] = A[i] + scalar * B[i]. Returns (gb_s, e_per_byte)."""
    scalar = 3.14159
    A = np.ones(n, dtype=np.float64) * 2.0
    B = np.ones(n, dtype=np.float64) * 1.5
    C = np.zeros(n, dtype=np.float64)

    # Warm up
    C[:] = A + scalar * B

    times, energies = [], []
    for _ in range(repeat):
        e0 = try_read_rapl()
        t0 = time.perf_counter()
        C[:] = A + scalar * B
        t1 = time.perf_counter()
        e1 = try_read_rapl()

        times.append(t1 - t0)
        if e0 is not None and e1 is not None:
            energies.append(e1 - e0)

    dt       = np.median(times)
    bytes_op = 3.0 * n * 8  # read A, read B, write C
    gb_s     = bytes_op / dt / 1e9

    e_per_byte = None
    if energies:
        e_j        = np.median(energies)
        e_per_byte = e_j / bytes_op if bytes_op > 0 else None

    return gb_s, e_per_byte, bytes_op, dt


def dot_product(n: int, repeat: int = 3):
    """Dot product: sum(X*Y). Returns (gflops_s, e_per_flop)."""
    X = np.random.randn(n).astype(np.float64) * 0.01
    Y = np.random.randn(n).astype(np.float64) * 0.01

    # Warm up
    _ = np.dot(X, Y)

    times, energies = [], []
    for _ in range(repeat):
        e0 = try_read_rapl()
        t0 = time.perf_counter()
        _ = np.dot(X, Y)
        t1 = time.perf_counter()
        e1 = try_read_rapl()

        times.append(t1 - t0)
        if e0 is not None and e1 is not None:
            energies.append(e1 - e0)

    dt         = np.median(times)
    flops      = 2.0 * n
    gflops_s   = flops / dt / 1e9

    e_per_flop = None
    if energies:
        e_j        = np.median(energies)
        e_per_flop = e_j / flops if flops > 0 else None

    return gflops_s, e_per_flop, flops, dt


def get_cpu_info():
    """Return dict with CPU name and core count."""
    info = {"name": "Unknown", "cores": os.cpu_count() or 1}
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["name"] = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL).decode().strip()
            info["name"] = out
    except Exception:
        pass
    return info


def get_gpu_info():
    """Return GPU name and power limit (W) if NVML available."""
    if not NVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name   = pynvml.nvmlDeviceGetName(handle).decode()
        limit  = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        return {"name": name, "power_limit_w": limit}
    except Exception:
        return None


def estimate_l3_size_bytes():
    """Estimate L3 cache size from /sys or lscpu."""
    try:
        result = subprocess.check_output(
            ["lscpu"], stderr=subprocess.DEVNULL).decode()
        for line in result.splitlines():
            if "L3" in line and "cache" in line.lower():
                # Parse e.g. "L3 cache: 40960K" or "40 MiB"
                parts = line.split(":")
                if len(parts) > 1:
                    val = parts[1].strip()
                    if "K" in val:
                        return float(val.replace("K", "").strip()) * 1024
                    if "M" in val or "MiB" in val:
                        return float(val.replace("MiB","").replace("M","").strip()) * 1024**2
    except Exception:
        pass
    return 40 * 1024**2  # 40 MB default


# ── Main calibration ────────────────────────────────────────────────────────

def calibrate(args):
    cpu     = get_cpu_info()
    gpu     = get_gpu_info()
    l3_size = estimate_l3_size_bytes()

    print(f"Platform: {cpu['name']}  ({cpu['cores']} cores)")
    print(f"L3 cache: {l3_size/1e6:.0f} MB")
    print(f"RAPL: {'available' if RAPL_AVAILABLE else 'NOT available (root required?)'}")
    print(f"NVML: {'available' if NVML_AVAILABLE else 'NOT available'}")
    if gpu:
        print(f"GPU:  {gpu['name']}  (TDP {gpu['power_limit_w']:.0f} W)")
    print()

    # ── STREAM Triad — choose array size > L3 to measure DRAM bandwidth ──
    n_dram = int(l3_size * 3 / 8)  # 3 × L3 size in doubles → forces DRAM
    n_l3   = int(l3_size * 0.3 / 8) # 0.3 × L3 → L3 bandwidth

    print(f"STREAM Triad (DRAM, n={n_dram:,})...")
    gb_dram, e_per_byte_dram, bytes_dram, dt_dram = stream_triad(n_dram, args.repeat)
    print(f"  Bandwidth: {gb_dram:.1f} GB/s  dt={dt_dram*1000:.2f} ms")
    if e_per_byte_dram is not None:
        print(f"  α_mem_dram = {e_per_byte_dram:.3e} J/byte  (RAPL)")
    else:
        # Estimate from power consumption and bandwidth
        idle_w = 50.0  # assumed
        e_per_byte_dram = idle_w / (gb_dram * 1e9) if gb_dram > 0 else 5e-9
        print(f"  α_mem_dram = {e_per_byte_dram:.3e} J/byte  (estimated from BW)")

    print(f"\nSTREAM Triad (L3, n={n_l3:,})...")
    gb_l3, e_per_byte_l3, _, _ = stream_triad(n_l3, args.repeat)
    print(f"  Bandwidth: {gb_l3:.1f} GB/s")
    if e_per_byte_l3 is not None:
        print(f"  α_mem_l3 = {e_per_byte_l3:.3e} J/byte  (RAPL)")
    else:
        e_per_byte_l3 = e_per_byte_dram * 0.1  # L3 ~10× cheaper than DRAM
        print(f"  α_mem_l3 = {e_per_byte_l3:.3e} J/byte  (estimated)")

    # ── Dot product FLOP benchmark ─────────────────────────────────────────
    n_dot = n_dram // 2  # memory-bound at this size
    print(f"\nDot product benchmark (n={n_dot:,})...")
    gflops, e_per_flop, flops, dt_dot = dot_product(n_dot, args.repeat)
    print(f"  GFLOP/s: {gflops:.2f}")
    if e_per_flop is not None:
        print(f"  α_flop = {e_per_flop:.3e} J/FLOP  (RAPL)")
    else:
        # Memory-bound: E_dot ≈ 2*n_dot*8 * α_mem_dram
        e_per_flop = 2 * n_dot * 8 * e_per_byte_dram / flops
        print(f"  α_flop = {e_per_flop:.3e} J/FLOP  (derived from memory model)")

    # ── SpMV-specific FLOP coefficient (30% overhead vs pure dot) ─────────
    alpha_flop_spmv  = e_per_flop * 1.3
    alpha_flop_blas1 = e_per_flop

    # ── Build output JSON ──────────────────────────────────────────────────
    coeffs = {
        "platform_name":    cpu["name"],
        "calibrated":       True,
        "rapl_used":        RAPL_AVAILABLE,
        "nvml_available":   NVML_AVAILABLE,
        "alpha_flop_spmv":  alpha_flop_spmv,
        "alpha_flop_blas1": alpha_flop_blas1,
        "alpha_flop_amg":   alpha_flop_spmv * 1.2,  # AMG slightly more overhead
        "alpha_mem_l1":     e_per_byte_l3 * 0.1,
        "alpha_mem_l2":     e_per_byte_l3 * 0.5,
        "alpha_mem_l3":     e_per_byte_l3,
        "alpha_mem_dram":   e_per_byte_dram,
        "alpha_comm":       e_per_byte_dram * 2.0,  # network ~ 2× DRAM energy/byte
        "idle_watts":       50.0,
        "peak_gflops_s":    gflops,
        "peak_gb_s_dram":   gb_dram,
        "peak_gb_s_l3":     gb_l3,
        "l3_size_bytes":    l3_size,
        "l2_size_bytes":    2 * 1024**2,
    }
    if gpu:
        coeffs["gpu_name"]         = gpu["name"]
        coeffs["gpu_power_limit_w"]= gpu["power_limit_w"]

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n=== Calibrated Energy Coefficients ===")
    print(f"  α_mem_dram   = {coeffs['alpha_mem_dram']:.3e} J/byte")
    print(f"  α_mem_l3     = {coeffs['alpha_mem_l3']:.3e} J/byte")
    print(f"  α_flop_spmv  = {coeffs['alpha_flop_spmv']:.3e} J/FLOP")
    print(f"  α_flop_blas1 = {coeffs['alpha_flop_blas1']:.3e} J/FLOP")
    print(f"  peak BW DRAM = {gb_dram:.1f} GB/s")
    print(f"  peak GFLOP/s = {gflops:.2f}")

    # ── Write JSON ─────────────────────────────────────────────────────────
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(coeffs, f, indent=2)
    print(f"\nWritten → {output}")
    print("\nIn C++ load with:")
    print(f'  auto coeffs = RooflineCoeffs::from_json("{output}");')
    print( '  RooflineModel model(coeffs);')
    return coeffs


def main():
    parser = argparse.ArgumentParser(description="Calibrate HAMLSS energy model")
    parser.add_argument("--output",   default="models/energy_coeffs.json")
    parser.add_argument("--repeat",   type=int, default=5,
                        help="Benchmark repetitions (more = more stable)")
    parser.add_argument("--platform", default="auto",
                        choices=["auto", "desktop", "server"],
                        help="Use pre-set defaults instead of measuring")
    args = parser.parse_args()

    if args.platform in ("desktop", "server") and not RAPL_AVAILABLE:
        # Use pre-set defaults
        from pathlib import Path
        import json
        defaults = {
            "desktop": {
                "platform_name":   "Desktop CPU (pre-set)",
                "calibrated":      False,
                "alpha_flop_spmv": 2.0e-10,
                "alpha_mem_dram":  5.0e-9,
                "alpha_mem_l3":    5.0e-10,
                "peak_gb_s_dram":  50.0,
                "peak_gflops_s":   100.0,
                "l3_size_bytes":   40e6,
                "idle_watts":      50.0,
            },
            "server": {
                "platform_name":   "Server CPU (pre-set)",
                "calibrated":      False,
                "alpha_flop_spmv": 2.5e-10,
                "alpha_mem_dram":  6.0e-9,
                "alpha_mem_l3":    6.0e-10,
                "peak_gb_s_dram":  100.0,
                "peak_gflops_s":   200.0,
                "l3_size_bytes":   80e6,
                "idle_watts":      80.0,
            },
        }
        preset = defaults[args.platform]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(preset, f, indent=2)
        print(f"Written pre-set defaults for '{args.platform}' → {args.output}")
        return

    calibrate(args)


if __name__ == "__main__":
    main()
