#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK_ERROR(code)                                                 \
  {                                                                            \
    if (code != cudaSuccess) {                                                 \
      printf("Error %d: %s in %s at line %d\n", code,                          \
             cudaGetErrorString(code), __FILE__, __LINE__);                    \
      exit(code);                                                              \
    }                                                                          \
  }

__global__ void vecAddKernel(float *A, float *B, float *C, float n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

void vecAdd(float *h_A, float *h_B, float *h_C, float n) {
  int THREADS_PER_BLOCK = 256;
  float *d_A, *d_B, *d_C;
  int totalBytes = n * sizeof(float);

  // Allocate memory for each vector on GPU, and copy vectors from host to GPU
  CUDA_CHECK_ERROR(cudaMalloc(&d_A, totalBytes));
  CUDA_CHECK_ERROR(cudaMalloc(&d_B, totalBytes));
  CUDA_CHECK_ERROR(cudaMalloc(&d_C, totalBytes));

  CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, totalBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, totalBytes, cudaMemcpyHostToDevice));

  // Invoke kernel
  int blockSize = THREADS_PER_BLOCK;
  int gridSize = (int)ceil((float)n / blockSize);

  vecAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

  // Check for any errors launching the kernel
  cudaError_t cudaStatus = cudaGetLastError();
  CUDA_CHECK_ERROR(cudaStatus);

  // Copy result from GPU to host
  CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, totalBytes, cudaMemcpyDeviceToHost));

  // Free memory on GPU
  CUDA_CHECK_ERROR(cudaFree(d_A));
  CUDA_CHECK_ERROR(cudaFree(d_B));
  CUDA_CHECK_ERROR(cudaFree(d_C));
}

bool checkResult(float *h_A, float *h_B, float *h_C, int n) {
  for (int i = 0; i < n; i++) {
    const float EPSILON = 1e-7;
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > EPSILON) {
      printf("h_A[%d] + h_B[%d] != h_C[%d]\n", i, i, i);
      printf("h_A[%d] = %f\n", i, h_A[i]);
      printf("h_B[%d] = %f\n", i, h_B[i]);
      printf("h_C[%d] = %f\n", i, h_C[i]);
      return false;
    }
  }
  return true;
}

int main() {
  float *h_A, *h_B, *h_C;
  int n = 1 << 16;
  int totalBytes = n * sizeof(float);

  h_A = (float *)malloc(totalBytes);
  h_B = (float *)malloc(totalBytes);
  h_C = (float *)malloc(totalBytes);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    printf("Error: malloc unable to allocate memory on host\n");
    exit(-1);
  }

  for (int i = 0; i < n; i++) {
    h_A[i] = ((((float)rand() / (float)(RAND_MAX)) * 300));
    h_B[i] = ((((float)rand() / (float)(RAND_MAX)) * 300));
  }

  vecAdd(h_A, h_B, h_C, n);

  if (!checkResult(h_A, h_B, h_C, n)) {
    printf("Incorrect result!\n");
  } else {
    printf("Correct result!\n");
  }

  free(h_A);
  free(h_B);
  free(h_C);
}