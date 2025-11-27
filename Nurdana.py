import numpy as np
from numba import cuda, float32, int32
import math
import time

# ======================= GPU kernels =======================

@cuda.jit
def dot_kernel_3x3(K, A, dotMats):
    """
    Each block = one 3x3 window in A.
    Each thread (0..8) = one element inside that 3x3 window.
    dotMats has shape (4*9,) : 4 windows, each 3x3 -> 9 products.
    """
    win = cuda.blockIdx.x       # 0..3 (4 windows)
    tid = cuda.threadIdx.x      # 0..8

    if tid >= 9:
        return

    # windows positions: (0,0), (0,1), (1,0), (1,1)
    wy = win // 2   # row of window
    wx = win % 2    # col of window

    ky = tid // 3   # row in kernel/window
    kx = tid % 3    # col in kernel/window

    # A is 4x4, K is 3x3, stored row-major
    a_val = A[(wy + ky) * 4 + (wx + kx)]
    k_val = K[ky * 3 + kx]

    dotMats[win * 9 + tid] = float32(a_val * k_val)


@cuda.jit
def reduce7_kernel(g_idata, g_odata, n):
    """
    Variant-7 style reduction (Mark Harris style) specialized for one block.
    - two elements per thread (grid-stride)
    - reduction in shared memory
    Here we use blockDim=16, n=9 (for each 3x3 window).
    """
    # shared memory, sized for up to blockDim.x elements
    smem = cuda.shared.array(64, dtype=float32)  # enough for blockDim<=64

    tid = cuda.threadIdx.x
    blockSize = cuda.blockDim.x

    i = cuda.blockIdx.x * (blockSize * 2) + tid
    gridSize = blockSize * 2 * cuda.gridDim.x

    my_sum = 0.0

    # grid-stride loop: 2 elements per thread per iteration
    while i < n:
        my_sum += g_idata[i]
        if i + blockSize < n:
            my_sum += g_idata[i + blockSize]
        i += gridSize

    smem[tid] = my_sum
    cuda.syncthreads()

    # reduction in shared memory (blockSize = 16 here)
    if blockSize >= 32:
        if tid < 16:
            smem[tid] += smem[tid + 16]
        cuda.syncthreads()
    if blockSize >= 16:
        if tid < 8:
            smem[tid] += smem[tid + 8]
        cuda.syncthreads()
    if blockSize >= 8:
        if tid < 4:
            smem[tid] += smem[tid + 4]
        cuda.syncthreads()
    if blockSize >= 4:
        if tid < 2:
            smem[tid] += smem[tid + 2]
        cuda.syncthreads()
    if blockSize >= 2:
        if tid < 1:
            smem[tid] += smem[tid + 1]
        cuda.syncthreads()

    if tid == 0:
        g_odata[0] = smem[0]


# ======================= CPU helpers =======================

def conv_cpu(K, A):
    """
    Reference CPU convolution:
    K: 3x3 (flat len 9)
    A: 4x4 (flat len 16)
    returns Result 2x2 (flat len 4)
    """
    Result = np.zeros(4, dtype=np.float32)
    for wy in range(2):
        for wx in range(2):
            s = 0.0
            for ky in range(3):
                for kx in range(3):
                    a_val = A[(wy + ky) * 4 + (wx + kx)]
                    k_val = K[ky * 3 + kx]
                    s += a_val * k_val
            Result[wy * 2 + wx] = s
    return Result


def print_matrix_int(arr, rows, cols):
    for r in range(rows):
        for c in range(cols):
            print(int(arr[r * cols + c]), end=" ")
        print()


def print_matrix_float(arr, rows, cols):
    for r in range(rows):
        for c in range(cols):
            print(f"{arr[r * cols + c]:.3f}", end=" ")
        print()


# ======================= Main =======================

def main():
    # Random seed for reproducibility (optional)
    np.random.seed(int(time.time()))

    # K: 3x3 in [-3, 3], A: 4x4 in [-4, 4]
    h_K = np.random.randint(-3, 4, size=(9,), dtype=np.int32)
    h_A = np.random.randint(-4, 5, size=(16,), dtype=np.int32)

    # Device arrays
    d_K = cuda.to_device(h_K)
    d_A = cuda.to_device(h_A)

    # 4 windows, each 9 products
    h_dotMats = np.zeros(4 * 9, dtype=np.float32)
    d_dotMats = cuda.device_array_like(h_dotMats)

    # Launch dot product kernel: 4 blocks (for 4 windows), 9 threads per block
    dot_kernel_3x3[4, 9](d_K, d_A, d_dotMats)
    cuda.synchronize()

    # Copy dot matrices back so we can print them later
    h_dotMats = d_dotMats.copy_to_host()

    # Now reduce each 3x3 block with variant-7 style kernel
    block_size = 16  # power of two >= 9
    h_sums = np.zeros(4, dtype=np.float32)

    for w in range(4):
        # slice for this window
        window_slice = d_dotMats[w * 9:(w + 1) * 9]
        d_sum = cuda.device_array(1, dtype=np.float32)

        shared_bytes = block_size * np.dtype(np.float32).itemsize
        reduce7_kernel[1, block_size, shared_bytes](window_slice, d_sum, 9)
        cuda.synchronize()

        h_sums[w] = d_sum.copy_to_host()[0]

    # Build GPU result (2x2) from sums
    h_ResultGPU = np.zeros(4, dtype=np.float32)
    for w in range(4):
        wy = w // 2
        wx = w % 2
        h_ResultGPU[wy * 2 + wx] = h_sums[w]

    # CPU reference
    h_ResultCPU = conv_cpu(h_K, h_A)

    # (optional) verification
    if not np.allclose(h_ResultGPU, h_ResultCPU, atol=1e-4):
        print("WARNING: GPU and CPU results differ!")
        # print("GPU:", h_ResultGPU)
        # print("CPU:", h_ResultCPU)

    # ======================= Output (exact order) =======================
    # K (3x3)
    print_matrix_int(h_K, 3, 3)
    print()

    # A (4x4)
    print_matrix_int(h_A, 4, 4)
    print()

    # Each dot multiplication matrix (4 windows × 3x3)
    for w in range(4):
        print_matrix_float(h_dotMats[w * 9:(w + 1) * 9], 3, 3)
        print()

    # Each sum (one per window)
    for w in range(4):
        print(f"{h_sums[w]:.3f}")
    print()

    # Result matrix (2x2)
    print_matrix_float(h_ResultGPU, 2, 2)


if __name__ == "__main__":
    main()
