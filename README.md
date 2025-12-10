# Exp - 1    GPU-based-vector-summation
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution confi guration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## Aim:
To compare CUDA vector addition performance using block.x = 1024 and block.x = 1023.
To modify the kernel so each thread processes 2 elements with block.x = 256 and compare performance.

## Procedure:
- Initialize two vectors of size 2 power 24.

- Compute CPU reference output.

- Run GPU kernel with:

- Case 1: block.x = 1024

- Case 2: block.x = 1023

- Modify kernel to process 2 elements per thread, run with block.x = 256.

- Measure execution time and verify results.

## Program:
```py
!nvidia-smi

%%writefile sumArraysOnGPU-timer.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <math.h>

#define CHECK(call)                                                        \
{                                                                          \
    const cudaError_t err = call;                                          \
    if (err != cudaSuccess)                                                \
    {                                                                      \
        printf("CUDA error %d at %s:%d: %s\n",                             \
               err, __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                           \
    }                                                                      \
}

double seconds()
{
    using namespace std::chrono;
    return duration<double>(high_resolution_clock::now()
            .time_since_epoch()).count();
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double eps = 1e-8;
    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > eps)
        {
            printf("Mismatch at %d: host=%f gpu=%f\n", i, hostRef[i], gpuRef[i]);
            return;
        }
    }
    printf("Arrays match.\n\n");
}

void initialData(float *ip, int size)
{
    srand(0);
    for (int i = 0; i < size; i++)
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumArraysOnHost(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumArraysOnGPU_two(float *A, float *B, float *C, const int N)
{
    int base = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int i1 = base;
    int i2 = base + blockDim.x;
    if (i1 < N) C[i1] = A[i1] + B[i1];
    if (i2 < N) C[i2] = A[i2] + B[i2];
}

int main()
{
    printf("sumArraysOnGPU-timer.cu starting...\n");

    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Using Device %d: %s\n", dev, prop.name);
    CHECK(cudaSetDevice(dev));

    int N = 1 << 24;
    size_t nBytes = N * sizeof(float);

    printf("Vector size: %d elements (%.2f MB)\n",
           N, nBytes / (1024.0 * 1024.0));

    float *h_A = (float*)malloc(nBytes);
    float *h_B = (float*)malloc(nBytes);
    float *hostRef = (float*)malloc(nBytes);
    float *gpuRef = (float*)malloc(nBytes);

    double tStart, tElaps;

    tStart = seconds();
    initialData(h_A, N);
    initialData(h_B, N);
    tElaps = seconds() - tStart;
    printf("initialData = %f sec\n", tElaps);

    tStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, N);
    tElaps = seconds() - tStart;
    printf("sumArraysOnHost = %f sec\n\n", tElaps);

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, nBytes));
    CHECK(cudaMalloc(&d_B, nBytes));
    CHECK(cudaMalloc(&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // ---------------------------------------------------------------
    // (i) block.x = 1024
    // ---------------------------------------------------------------
    printf("--- Case 1: block.x = 1024 ---\n");
    int block1 = 1024;
    int grid1  = (N + block1 - 1) / block1;
    CHECK(cudaMemset(d_C, 0, nBytes));

    tStart = seconds();
    sumArraysOnGPU<<<grid1, block1>>>(d_A, d_B, d_C, N);
    CHECK(cudaDeviceSynchronize());
    tElaps = seconds() - tStart;
    printf("Time (1024): %f sec\n", tElaps);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, N);

    // ---------------------------------------------------------------
    // (i) block.x = 1023
    // ---------------------------------------------------------------
    printf("--- Case 2: block.x = 1023 ---\n");
    int block2 = 1023;
    int grid2  = (N + block2 - 1) / block2;
    CHECK(cudaMemset(d_C, 0, nBytes));

    tStart = seconds();
    sumArraysOnGPU<<<grid2, block2>>>(d_A, d_B, d_C, N);
    CHECK(cudaDeviceSynchronize());
    tElaps = seconds() - tStart;
    printf("Time (1023): %f sec\n", tElaps);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, N);

    // ---------------------------------------------------------------
    // (ii) block.x = 256 with 2 elements per thread
    // ---------------------------------------------------------------
    printf("--- Case 3: 256 threads, 2 elements/thread ---\n");
    int block3 = 256;
    int grid3  = (N + block3 * 2 - 1) / (block3 * 2);
    CHECK(cudaMemset(d_C, 0, nBytes));

    tStart = seconds();
    sumArraysOnGPU_two<<<grid3, block3>>>(d_A, d_B, d_C, N);
    CHECK(cudaDeviceSynchronize());
    tElaps = seconds() - tStart;
    printf("Time (256, 2 elements): %f sec\n", tElaps);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}

!nvcc -arch=sm_75 sumArraysOnGPU-timer.cu -o sumGPU

!./sumGPU

```

## Output:
<img width="440" height="359" alt="image" src="https://github.com/user-attachments/assets/9aa0222a-bb33-4654-ae31-db483d551b34" />

## Result:
The 256-thread kernel with 2 elements per thread gave the fastest execution time, followed by 1023 and 1024 threads per block, with all cases producing matching results.
