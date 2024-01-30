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

__global__ void matrixAddElementKernel(float *out, float *A, float *B,
                                       float height, float width) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = width * y + x;
  out[idx] = A[idx] + B[idx];
}

__global__ void matrixAddRowKernel(float *out, float *A, float *B, float height,
                                   float width) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= height)
    return;

  for (int col = 0; col < width; col++) {
    int idx = row * width + col;
    out[idx] = A[idx] + B[idx];
  }
}

__global__ void matrixAddColKernel(float *out, float *A, float *B, float height,
                                   float width) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col >= width)
    return;

  for (int row = 0; row < height; row++) {
    int idx = row * width + col;
    out[idx] = A[idx] + B[idx];
  }
}

typedef enum { ROW, COL, ELEMENT } MatrixAddType;

void matrixAdd(float *h_A, float *h_B, float *h_C, float height, float width,
               MatrixAddType type = ELEMENT) {
  int THREADS_PER_BLOCKDIM = 16;
  float *d_A, *d_B, *d_C;
  int totalBytes = height * width * sizeof(float);

  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_A, totalBytes));
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_B, totalBytes));
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_C, totalBytes));

  CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, totalBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, totalBytes, cudaMemcpyHostToDevice));

  if (type == ROW) {
    dim3 threadsPerBlock(THREADS_PER_BLOCKDIM);
    dim3 blocksPerGrid(ceil(height / (float)THREADS_PER_BLOCKDIM));
    matrixAddRowKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B,
                                                           height, width);
  } else if (type == COL) {
    dim3 threadsPerBlock(THREADS_PER_BLOCKDIM);
    dim3 blocksPerGrid(ceil(width / (float)THREADS_PER_BLOCKDIM));
    matrixAddColKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B,
                                                           height, width);
  } else {
    dim3 threadsPerBlock(THREADS_PER_BLOCKDIM, THREADS_PER_BLOCKDIM);
    dim3 blocksPerGrid(ceil(width / (float)THREADS_PER_BLOCKDIM),
                       ceil(height / (float)THREADS_PER_BLOCKDIM));
    matrixAddElementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B,
                                                               height, width);
  }

  CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, totalBytes, cudaMemcpyDeviceToHost));
}

bool checkResult(float *h_A, float *h_B, float *h_C, int height, int width) {
  for (int i = 0; i < height * width; i++) {
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
  float *h1_A, *h1_B, *h1_C, *h2_A, *h2_B, *h2_C, *h3_A, *h3_B, *h3_C;
  int width = 1 << 10;
  int height = 1 << 10;

  int totalBytes = width * height * sizeof(float);

  h1_A = (float *)malloc(totalBytes);
  h1_B = (float *)malloc(totalBytes);
  h1_C = (float *)malloc(totalBytes);
  h2_A = (float *)malloc(totalBytes);
  h2_B = (float *)malloc(totalBytes);
  h2_C = (float *)malloc(totalBytes);
  h3_A = (float *)malloc(totalBytes);
  h3_B = (float *)malloc(totalBytes);
  h3_C = (float *)malloc(totalBytes);

  for (int i = 0; i < width * height; i++) {
    float a = rand() % 100;
    float b = rand() % 100;
    h1_A[i] = a;
    h1_B[i] = b;
    h2_A[i] = a;
    h2_B[i] = b;
    h3_A[i] = a;
    h3_B[i] = b;
  }

  matrixAdd(h1_A, h1_B, h1_C, height, width, ELEMENT);
  matrixAdd(h2_A, h2_B, h2_C, height, width, ROW);
  matrixAdd(h3_A, h3_B, h3_C, height, width, COL);

  if (checkResult(h1_A, h1_B, h1_C, height, width)) {
    printf("Element-wise matrix addition is correct\n");
  } else {
    printf("Element-wise matrix addition is incorrect\n");
  }

  if (checkResult(h2_A, h2_B, h2_C, height, width)) {
    printf("Row-wise matrix addition is correct\n");
  } else {
    printf("Row-wise matrix addition is incorrect\n");
  }

  if (checkResult(h3_A, h3_B, h3_C, height, width)) {
    printf("Column-wise matrix addition is correct\n");
  } else {
    printf("Column-wise matrix addition is incorrect\n");
  }
}