#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// Function to initialize matrices with random values
void init_matrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        matrix[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
}

// Kernel function for element-wise addition of matrices
__global__ void matrix_elementwise_add(float *a, float *b, float *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

// Function to print a matrix
void print_matrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    const int M = 1 << 10; // Number of rows we should manipulate this vector size to get benchmark. M=N
    const int N = 1 << 10; // Number of columns

    const int size = M * N * sizeof(float);

    float *a, *b, *c;    // Host matrices
    float *d_a, *d_b, *d_c; // Device matrices

    // Allocate memory for host matrices
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    // Initialize host matrices with random values
    init_matrix(a, M * N);
    init_matrix(b, M * N);

    // Allocate memory on device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy host matrices to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(256);    //also we should manipulate this block size to get benchmark and we are going to change grid size by changing this.
    dim3 gridDim((M * N + blockDim.x - 1) / blockDim.x);

    // Timing start
    clock_t start = clock();

    // Launch kernel
    matrix_elementwise_add<<<gridDim, blockDim>>>(d_a, d_b, d_c, M * N);

    // Timing end
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f ms\n", elapsed_time_ms);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print resulting matrix
    //printf("Resulting Matrix:\n");
    //print_matrix(c, M, N);

    // Cleanup
    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}