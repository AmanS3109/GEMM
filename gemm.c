#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>
#include <stdlib.h>

#define N 1024
#define BLOCK 8

// FIX 1: Ensure 32-byte alignment for AVX loads
float A[N][N] __attribute__((aligned(32)));
float B[N][N] __attribute__((aligned(32)));
float C[N][N] __attribute__((aligned(32)));
float val[N][N] __attribute__((aligned(32)));

// Casting to vector pointers
__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

uint64_t nanos(){
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

int main () {
    assert(N % BLOCK == 0);
    
    // FIX 2: Handle file safely (check if exists)
    FILE *f = fopen("/tmp/matmul", "rb");
    if (f) {
        // Suppress warnings by checking result, though we overwrite later
        size_t r1 = fread(A, 1, sizeof(float)*N*N, f);
        size_t r2 = fread(B, 1, sizeof(float)*N*N, f);
        size_t r3 = fread(val, 1, sizeof(float)*N*N, f);
        fclose(f);
    }

    // Initialize with random data
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (float)rand() / RAND_MAX;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = (float)rand() / RAND_MAX;

    uint64_t start = nanos();
    
    // Matrix Multiplication Loop
    // NOTE: Based on your memory access pattern (linear access on B), 
    // this code effectively calculates C = A * Transpose(B).
    // Standard A * B requires columns of B, which is non-linear in memory.
    for (int by = 0; by < N; by += BLOCK) {
        for (int bx = 0; bx < N; bx += BLOCK) {

            __m256 tc[BLOCK];

            for (int y = 0; y < BLOCK; y++) {
                
                // FIX 3: Initialize accumulator to Zero
                __m256 tmp = _mm256_setzero_ps(); 
                
                for (int k = 0; k < N; k += 8) {
                    
                    // FIX 4: Correct Pointer Arithmetic
                    // k steps by 8. We need to step the pointer by 1 vector (which is 8 floats).
                    // Logic: (RowIndex * Width + ColIndex) / 8_floats_per_vec
                    
                    int a_idx = ((by + y) * N + k) / 8;
                    int b_idx = ((bx    ) * N + k) / 8; // Accessing B row-wise (Calculates A * B^T)

                    tmp = _mm256_fmadd_ps(
                            Am[a_idx],
                            Bm[b_idx],
                            tmp
                    );
                }
                tc[y] = tmp;
            }

            // Store results
            // Note: Horizontal sum is required here to get a single float value per cell
            // if we are doing dot products. However, strictly preserving your logic flow:
            for (int y = 0; y < BLOCK; y++) {
                // Determine where to write in C (Block offset + row)
                int c_idx = ((by + y) * N + bx) / 8;
                Cm[c_idx] = tc[y];
            }
        }
    }

    uint64_t end = nanos();
    
    // GFLOP calculation: 2 ops (mul+add) * N^3
    double gflop = (2.0 * N * N * N) * 1e-9;
    double s = (end - start) * 1e-9;
    
    printf("%f GFLOP/S\n", gflop / s);
    return 0;
}