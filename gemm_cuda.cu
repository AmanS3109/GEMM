#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024

// ---------------- CUDA KERNEL ----------------
__global__
void gemm_kernel(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------------- MAIN ----------------
int main() {
    size_t bytes = N * N * sizeof(float);

    // Host memory
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);

    // Initialize input matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    // Device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    // Copy inputs to device
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    // ---------------- TIMING ----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Optional warm-up (important for stable timing)
    gemm_kernel<<<blocks, threads>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start);

    // Kernel launch
    gemm_kernel<<<blocks, threads>>>(dA, dB, dC);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // ---------------- GFLOPS ----------------
    double seconds = ms * 1e-3;
    double gflops = (2.0 * N * N * N) * 1e-9 / seconds;

    printf("CUDA GEMM time: %.3f ms\n", ms);
    printf("CUDA GEMM performance: %.2f GFLOPS\n", gflops);

    // Copy result back (optional correctness check)
    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);
    printf("C[0] = %f (expected %f)\n", C[0], (float)N);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);

    return 0;
}
