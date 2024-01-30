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

#define MALLOC_CHECK_ERROR(ptr)                                                \
  {                                                                            \
    if (ptr == NULL) {                                                         \
      printf("Error: malloc failed in %s at line %d\n", __FILE__, __LINE__);   \
      exit(-1);                                                                \
    }                                                                          \
  }

__global__ void matrixVectorMultKernel(float *vector, float *matrix, float *out,
                                       int size) { // size is now an int
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if (row >= size)
    return;

  float sum =
      0.0f; // Accum in a separate float instead of on out to save mem access
  for (int matCol = 0; matCol < size; matCol++) {
    sum += matrix[row * size + matCol] * vector[matCol];
  }

  out[row] = sum;
}

void matrixVectorMultiply(float *vector, float *matrix, float *out, int size) {
  int THREADS_PER_BLOCK = 256;
  int totalBytesVec = size * sizeof(float);
  int totalBytesMat = size * size * sizeof(float);

  float *d_vector, *d_matrix, *d_out;

  // Allocate memory for each vector on GPU, and copy vectors from host to GPU
  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_vector, totalBytesVec));
  CUDA_CHECK_ERROR(
      cudaMemcpy(d_vector, vector, totalBytesVec, cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_matrix, totalBytesMat));
  CUDA_CHECK_ERROR(
      cudaMemcpy(d_matrix, matrix, totalBytesMat, cudaMemcpyHostToDevice));

  CUDA_CHECK_ERROR(cudaMalloc((void **)&d_out, totalBytesVec));

  // Launch kernel on GPU
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = ceil((float)size / threadsPerBlock);

  matrixVectorMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vector, d_matrix,
                                                             d_out, size);

  // Copy result back to host
  CUDA_CHECK_ERROR(
      cudaMemcpy(out, d_out, totalBytesVec, cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK_ERROR(cudaFree(d_vector));
  CUDA_CHECK_ERROR(cudaFree(d_matrix));
  CUDA_CHECK_ERROR(cudaFree(d_out));
}

bool checkResult(float *vector, float *matrix, float *out, int size) {
  int EPSILON = 1e-5;
  for (int i = 0; i < size; i++) {
    float sum = 0.0f;
    for (int j = 0; j < size; j++) {
      sum += matrix[i * size + j] * vector[j];
    }
    if (fabs(out[i] - sum) > EPSILON) {
      printf("Error: out[%d] = %f != %f\n", i, out[i], sum);
      return false;
    }
  }
  return true;
}

int main() {

  float *matrix, *vector, *out;
  int n = 1 << 10; // 2^10
  int totalBytesVec = n * sizeof(float);
  int totalBytesMat = n * n * sizeof(float);

  matrix = (float *)malloc(totalBytesMat);
  MALLOC_CHECK_ERROR(matrix);

  vector = (float *)malloc(totalBytesVec);
  MALLOC_CHECK_ERROR(vector);

  out = (float *)malloc(totalBytesVec);
  MALLOC_CHECK_ERROR(out);

  for (int i = 0; i < n; i++) {
    vector[i] = 1.0f;
    for (int j = 0; j < n; j++) {
      matrix[i * n + j] = 1.0f;
    }
  }

  matrixVectorMultiply(vector, matrix, out, n);

  printf("Checking results...\n");
  if (checkResult(vector, matrix, out, n)) {
    printf("Success!\n");
  } else {
    printf("Failure!\n");
  }

  free(matrix);
  free(vector);
  free(out);
}