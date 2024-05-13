#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/utils.cuh"

__global__ void vec_add_kernel(float *A, float *B, float *C, float n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

void vec_add(float *h_A, float *h_B, float *h_C, float n) {
  sptr<float> d_A = cuda_ptr_from_host(h_A, n);
  sptr<float> d_B = cuda_ptr_from_host(h_B, n);
  sptr<float> d_C = cuda_ptr_from_host(h_C, n);

  int bs = THREADS_PER_BLOCK;
  int gs = (int)ceil((float)n / bs);

  vec_add_kernel<<<gs, bs>>>(d_A.get(), d_B.get(), d_C.get(), n);

  CUDA_SYNC_AND_CHECK();

  copy_to_host(h_C, d_C, n);
}


bool check_vec_add_result(float *h_A, float *h_B, float *h_C, int n) {
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
  int n = 1 << 16;

  sptr<float> h_A = host_alloc<float>(n);
  sptr<float> h_B = host_alloc<float>(n);
  sptr<float> h_C = host_alloc<float>(n);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    printf("Error: malloc unable to allocate memory on host\n");
    exit(-1);
  }

  for (int i = 0; i < n; i++) {
    h_A.get()[i] = ((((float)rand() / (float)(RAND_MAX)) * 300));
    h_B.get()[i] = ((((float)rand() / (float)(RAND_MAX)) * 300));
  }

  vec_add(h_A.get(), h_B.get(), h_C.get(), n);

  if (!check_vec_add_result(h_A.get(), h_B.get(), h_C.get(), n)) {
    printf("Incorrect result!\n");
  } else {
    printf("Correct result!\n");
  }
}