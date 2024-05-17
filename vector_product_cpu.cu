#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vector_inner_product(float* a, float* b, int n, float* result)
{
    *result = 0.0f;
    for (int i = 0; i < n; i++)
    {
        *result += a[i] * b[i];
    }
}

int main(void)
{
    const int N = 1 << 10; // Number of elements in arrays, we should manipulate this vector size to get benchmark. 
    float* a, * b;
    float inner_product = 0.0f;

    size_t size = N * sizeof(float);

    // Allocate memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);

    // Define sample inputs, I defined an initial condition by using i and N
    for (int i = 0; i < N; i++)
    {
        a[i] = i + 1;
        b[i] = N - i;
    }

    // Vector inner product on CPU
    clock_t start = clock();
    vector_inner_product(a, b, N, &inner_product);
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time elapsed for %d elements: %f ms\n", N, elapsed_time_ms);

    // Print inner product(optional)
    //printf("Inner Product: %f\n", inner_product);

    // cleanup the host memory
    free(a);
    free(b);

    return 0;
}