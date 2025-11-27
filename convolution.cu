#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t e = (err);                                                 \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ------------------------- Reduction (variant 7 style) -------------------------

template <unsigned int blockSize>
__global__ void reduce7(const float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float mySum = 0.0f;

    // fully coalesced loads with two elements per thread
    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n)
            mySum += g_idata[i + blockSize];
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // reduction in shared memory
    if (blockSize >= 1024) {
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    // unroll last warp, no __syncthreads() needed
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockSize >= 64)  smem[tid] += smem[tid + 32];
        if (blockSize >= 32)  smem[tid] += smem[tid + 16];
        if (blockSize >= 16)  smem[tid] += smem[tid + 8];
        if (blockSize >= 8)   smem[tid] += smem[tid + 4];
        if (blockSize >= 4)   smem[tid] += smem[tid + 2];
        if (blockSize >= 2)   smem[tid] += smem[tid + 1];
    }

    // write result for this block
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ------------------------- Dot product kernel (convolution window) -------------------------

// Computes element-wise products K ⊙ window(A) for each 3x3 window.
// Each block = one window; each thread (0..8) = one element of 3x3.
__global__ void dotKernel3x3(const int *K, const int *A, float *dotMats) {
    int win = blockIdx.x;      // 0..3 (because (4-3+1)^2 = 4 windows)
    int tid = threadIdx.x;     // 0..8

    if (tid >= 9) return;      // safety if blockDim > 9

    // window top-left coordinates in A
    int wy = win / 2;          // rows: 0..1
    int wx = win % 2;          // cols: 0..1

    // position inside 3x3 kernel/window
    int ky = tid / 3;
    int kx = tid % 3;

    int aVal = A[(wy + ky) * 4 + (wx + kx)];  // A is 4x4
    int kVal = K[ky * 3 + kx];                // K is 3x3

    dotMats[win * 9 + tid] = static_cast<float>(aVal * kVal);
}

// ------------------------- Host helpers -------------------------

void fillRandomInt(int *data, int n, int minVal, int maxVal) {
    for (int i = 0; i < n; ++i) {
        int r = rand() % (maxVal - minVal + 1) + minVal;
        data[i] = r;
    }
}

void printMatrixInt(const int *data, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%d ", data[r * cols + c]);
        }
        printf("\n");
    }
}

void printMatrixFloat(const float *data, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            printf("%.3f ", data[r * cols + c]);
        }
        printf("\n");
    }
}

// CPU reference convolution for correctness check
void convCPU(const int *K, const int *A, float *Result) {
    // K: 3x3, A: 4x4, Result: 2x2
    for (int wy = 0; wy < 2; ++wy) {
        for (int wx = 0; wx < 2; ++wx) {
            float sum = 0.0f;
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int aVal = A[(wy + ky) * 4 + (wx + kx)];
                    int kVal = K[ky * 3 + kx];
                    sum += static_cast<float>(aVal * kVal);
                }
            }
            Result[wy * 2 + wx] = sum;
        }
    }
}

bool verify(const float *gpu, const float *cpu, int n, float eps = 1e-4f) {
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(gpu[i] - cpu[i]);
        if (diff > eps) return false;
    }
    return true;
}

// ------------------------- Main -------------------------

int main() {
    srand(static_cast<unsigned int>(time(NULL)));

    // Host matrices
    int   h_K[9];    // 3x3
    int   h_A[16];   // 4x4
    float h_dotMats[4 * 9]; // 4 windows, each 3x3
    float h_sums[4];        // 4 sums (one per window)
    float h_ResultGPU[4];   // 2x2
    float h_ResultCPU[4];   // 2x2

    // Fill K and A with required ranges
    fillRandomInt(h_K, 9,  -3,  3);  // [-3, 3]
    fillRandomInt(h_A, 16, -4,  4);  // [-4, 4]

    // --- GPU part: convolution via dot products + reduction ---

    int *d_K = nullptr;
    int *d_A = nullptr;
    float *d_dotMats = nullptr;
    float *d_sums = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_K, 9 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_A, 16 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_dotMats, 4 * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_sums, 4 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_K, h_K, 9 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, 16 * sizeof(int), cudaMemcpyHostToDevice));

    // 4 windows, each handled by one block
    dim3 gridDot(4);
    dim3 blockDot(9); // 9 threads per block (one per kernel/window element)
    dotKernel3x3<<<gridDot, blockDot>>>(d_K, d_A, d_dotMats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reduction: use variant-7 style kernel for each 3x3 dot matrix
    const unsigned int blockSize = 16;  // power of two >= 9
    for (int w = 0; w < 4; ++w) {
        const float *windowData = d_dotMats + w * 9;
        float *outPtr = d_sums + w;
        reduce7<blockSize><<<1, blockSize, blockSize * sizeof(float)>>>(windowData, outPtr, 9);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back dot matrices & sums
    CUDA_CHECK(cudaMemcpy(h_dotMats, d_dotMats,
                          4 * 9 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums,
                          4 * sizeof(float), cudaMemcpyDeviceToHost));

    // Build GPU Result (2x2) from sums (each window → one output)
    for (int w = 0; w < 4; ++w) {
        int wy = w / 2;
        int wx = w % 2;
        h_ResultGPU[wy * 2 + wx] = h_sums[w];
    }

    // --- CPU reference ---
    convCPU(h_K, h_A, h_ResultCPU);

    // Optional check (you can remove this print in final exam submission)
    // if (!verify(h_ResultGPU, h_ResultCPU, 4)) {
    //     fprintf(stderr, "Verification FAILED\n");
    // } else {
    //     fprintf(stderr, "Verification OK\n");
    // }

    // ----------------- Output (no extra messages) -----------------
    // Order required by the statement:
    // K, A, each dot multiplication matrix, each sum, Result matrix

    // K (3x3)
    printMatrixInt(h_K, 3, 3);
    printf("\n");

    // A (4x4)
    printMatrixInt(h_A, 4, 4);
    printf("\n");

    // Each of the dot multiplication matrices (4 windows, 3x3 each)
    for (int w = 0; w < 4; ++w) {
        printMatrixFloat(h_dotMats + w * 9, 3, 3);
        printf("\n");
    }

    // Each sum (one per window)
    for (int w = 0; w < 4; ++w) {
        printf("%.3f\n", h_sums[w]);
    }
    printf("\n");

    // Result matrix (2x2)
    printMatrixFloat(h_ResultGPU, 2, 2);

    // Cleanup
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_dotMats));
    CUDA_CHECK(cudaFree(d_sums));

    return 0;
}
