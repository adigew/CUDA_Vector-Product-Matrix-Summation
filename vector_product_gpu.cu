#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

void vector_inner_product(float* a, float* b, float* out, int n)
{
    float sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }
    *out = sum;
}

__global__ void vector_inner_product_kernel(float* a, float* b, float* out, int N)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    shared[tid] = (i < N) ? (a[i] * b[i]) : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        out[blockIdx.x] = shared[0];
    }
}

int main(void)
{
    const int N = 1 << 20; // Number of elements in arrays,we should manipulate this vector size to get benchmark.
    float* a, * b, * out;
    float* a_d, * b_d, * out_d;

    int BLOCK_SIZE = 256; // also we should manipulate this block size to get benchmark and we are going to change grid size by changing this.
    int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 BLOCK_SIZE_DIM3 = dim3(BLOCK_SIZE, 1, 1);
    dim3 NUM_BLOCKS_DIM3 = dim3(NUM_BLOCKS, 1, 1);

    size_t size = N * sizeof(float);

    // Allocate memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    out = (float*)malloc(NUM_BLOCKS * sizeof(float));

    cudaMalloc((void**)&a_d, size);
    cudaMalloc((void**)&b_d, size);
    cudaMalloc((void**)&out_d, NUM_BLOCKS * sizeof(float));

    // Define sample inputs
    for (int i = 0; i < N; i++)
    {
        a[i] = i + 1;
        b[i] = N - i;
    }

    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

    // Vector inner product on GPU
    clock_t start = clock();

    vector_inner_product_kernel << <NUM_BLOCKS_DIM3, BLOCK_SIZE_DIM3, BLOCK_SIZE * sizeof(float) >> > (a_d, b_d, out_d, N);
    cudaDeviceSynchronize();

    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed for %d elements: %f ms\n", N, elapsed_time_ms);

    cudaMemcpy(out, out_d, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate final inner product on CPU (optional)
    //float final_result = 0.0;
    //for (int i = 0; i < NUM_BLOCKS; i++)
    //{
    //    final_result += out[i];
    //}

    //printf("Inner product of the vectors: %f\n", final_result);

    // cleanup the host memory
    free(a);
    free(b);
    free(out);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);
}