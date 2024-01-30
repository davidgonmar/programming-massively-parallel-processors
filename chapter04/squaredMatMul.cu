#include "cuda.h"
#include <iostream>

#define CUDA_CHECK_ERROR(code)                                                 \
  {                                                                            \
    if (code != cudaSuccess) {                                                 \
      printf("Error %d: %s in %s at line %d\n", code,                          \
             cudaGetErrorString(code), __FILE__, __LINE__);                    \
      exit(code);                                                              \
    }                                                                          \
  }


__global__ void squaredMatMulKernel(float *A, float *B, float *out, int size) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= size || col >= size) return;

  float accum = 0.0f;
  for (int idx = 0; idx < size; idx++) {
    accum += A[row * size + idx] * B[idx * size + col];
  }
  out[row * size + col] = accum;
}

void squaredMatMul(float *A, float *B, float *out, int size) {
  float *d_A, *d_B, *d_out;

  int numBytes = size * size * sizeof(float);

  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_A, numBytes));
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_B, numBytes));
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_out, numBytes));

  CUDA_CHECK_ERROR(cudaMemcpy(d_A, A, numBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK_ERROR(cudaMemcpy(d_B, B, numBytes, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);

  squaredMatMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_out, size);

  CUDA_CHECK_ERROR(cudaMemcpy(out, d_out, numBytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK_ERROR(cudaFree(d_A));
  CUDA_CHECK_ERROR(cudaFree(d_B));
  CUDA_CHECK_ERROR(cudaFree(d_out));
}

bool checkResult(float *A, float *B, float *out, int size) {
  for (int row = 0; row < size; row++) {
    for (int col = 0; col < size; col++) {
      float accum = 0.0f;
      for (int idx = 0; idx < size; idx++) {
        accum += A[row * size + idx] * B[idx * size + col];
      }
      if (accum != out[row * size + col]) {
        std::cout << "Error: Expected[" << row << "][" << col << "] = " << accum
                  << " but got " << out[row * size + col] << std::endl;
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char **argv) {

  int size = 1024;
  float *A = new float[size * size];
  float *B = new float[size * size];
  float *out = new float[size * size];

  for (int i = 0; i < size * size; i++) {
    A[i] = 1.0f;
    B[i] = 1.0f;
  }

  squaredMatMul(A, B, out, size);

  if (checkResult(A, B, out, size)) {
    std::cout << "Success!" << std::endl;
  } else {
    std::cout << "Failure!" << std::endl;
  }

  delete[] A;
  delete[] B;
  delete[] out;

  return 0;
  
}