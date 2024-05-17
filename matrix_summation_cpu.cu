#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to initialize matrices with random values
void init_matrix(float *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        matrix[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
}

// Function for element-wise addition of matrices
void matrix_elementwise_add(float *a, float *b, float *c, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            c[i * cols + j] = a[i * cols + j] + b[i * cols + j];
        }
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

    const int size = M * N;

    float *a, *b, *c; // Host matrices

    // Allocate memory for host matrices
    a = (float *)malloc(size * sizeof(float));
    b = (float *)malloc(size * sizeof(float));
    c = (float *)malloc(size * sizeof(float));

    // Initialize host matrices with random values
    init_matrix(a, size);
    init_matrix(b, size);

    // Timing start
    clock_t start = clock();

    // Perform element-wise addition of matrices
    matrix_elementwise_add(a, b, c, M, N);

    // Timing end
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed: %f ms\n", elapsed_time_ms);

    // Print resulting matrix
    //printf("Resulting Matrix:\n");
    //print_matrix(c, M, N);

    // Cleanup
    free(a);
    free(b);
    free(c);

    return 0;
}