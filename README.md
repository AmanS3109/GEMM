# GEMM Benchmarking: CPU (AVX), CUDA, and NumPy

This repository is an educational and experimental project focused on **understanding and benchmarking General Matrix Multiplication (GEMM)** across different computation platforms:

- CPU (C with AVX intrinsics)
- GPU (CUDA C)
- Python (NumPy using BLAS)

The goal is **not just performance**, but a **deep understanding of how hardware, memory hierarchy, and execution models affect GEMM performance**.

---

## Project Motivation

Matrix multiplication looks simple mathematically, but achieving high performance is extremely difficult due to:

- Memory hierarchy (registers, caches, RAM)
- Data reuse and cache locality
- SIMD vs SIMT execution models
- CPU vs GPU architectural differences

This project explores GEMM step by step, starting from low-level CPU code and gradually moving to GPU execution.

---

## Implementations

### 1. CPU GEMM (C + AVX)

- Language: C
- SIMD: AVX (256-bit, `__m256`)
- Technique:
  - Register blocking
  - FMA (`_mm256_fmadd_ps`)
  - Aligned global matrices
- Single-threaded (no OpenMP yet)
- Focus: understanding registers, alignment, and memory access

### GPU GEMM (CUDA C)

- Language: CUDA C
- Model: SIMT (CUDA threads)
- Baseline implementation:
  - One thread computes one `C[row][col]`
  - No shared memory tiling yet
  - No cuBLAS (manual kernel)
- Kernel-only timing using CUDA events
- Focus: understanding GPU execution, threads, blocks, and timing

### Python GEMM (NumPy)

- Language: Python
- Backend: NumPy → BLAS (MKL / OpenBLAS)
- Highly optimized, multi-threaded, production-grade GEMM
- Used as a **reference ceiling**, not as handwritten code

---

## Benchmark Setup

- Matrix size: `1024 × 1024`
- Data type: `float32`
- Operation: `C = A × B`


- Timing:
- CPU: `clock_gettime(CLOCK_MONOTONIC_RAW)`
- CUDA: `cudaEventRecord`
- NumPy: repeated iterations, best performance reported

---

## Current Benchmark Results (as of today)

| Implementation | Performance |
|----------------|-------------|
| **CPU (C + AVX)** | **146.32 GFLOP/s** |
| **CUDA (naive kernel)** | **548.75 GFLOPS** |
| **Python (NumPy, max over 20,000 iters)** | **~581 GFLOPS** |

> ⚠️ Notes:
> - CUDA result is from a **naive kernel**, without shared memory tiling.
> - NumPy performance comes from highly optimized BLAS libraries.
> - CPU implementation is single-threaded and not cache-blocked yet.

---

## Key Observations So Far

- Even a **naive CUDA kernel** can outperform a well-written single-core AVX CPU implementation.
- NumPy achieves near-GPU-level performance because it uses **highly optimized BLAS (MKL/OpenBLAS)**.
- Raw FLOPS are not just about math — **memory movement dominates performance**.
- CPU and GPU require **completely different optimization strategies**.

---

## What This Project Is (and Is Not)

### ✔ This project is:
- A learning-oriented GEMM exploration
- A comparison of CPU SIMD vs GPU SIMT
- A deep dive into memory hierarchy and performance

### ✘ This project is NOT:
- A replacement for BLAS or cuBLAS
- A claim of beating vendor-optimized libraries
- A production GEMM implementation (yet)

---

## Planned Next Steps

### CPU Side
- Cache blocking (L1/L2 aware tiling)
- Multi-threading with OpenMP
- Fair comparison with MKL single-thread vs multi-thread

### CUDA Side
- Shared memory tiling
- Register blocking inside kernels
- Occupancy analysis
- Comparison with cuBLAS

### Analysis
- Roofline model comparison
- Arithmetic intensity calculations
- Memory bandwidth vs compute limits

---
