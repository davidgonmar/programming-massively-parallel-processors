#pragma once
#include <memory>
#include <stdexcept>
#include <string>

#define CUDA_CHECK_ERROR(code)                                                 \
  {                                                                            \
    if (code != cudaSuccess) {                                                 \
      std::string error_message = "Error " + std::to_string(code) + ": " +     \
                                  cudaGetErrorString(code) + " in " + __FILE__ + \
                                  " at line " + std::to_string(__LINE__);        \
      throw std::runtime_error(error_message);                                \
    }                                                                          \
  }

#define CUDA_SYNC_AND_CHECK() CUDA_CHECK_ERROR(cudaDeviceSynchronize())

template <typename T>
bool check_equal(T *A, T *B, int size) {
  for (int i = 0; i < size; i++) {
    const float EPSILON = 1e-7;
    if (fabs(A[i] - B[i]) > EPSILON) {
      printf("A[%d] != B[%d]\n", i, i);
      printf("A[%d] = %f\n", i, A[i]);
      printf("B[%d] = %f\n", i, B[i]);
      return false;
    }
  }
  return true;
}

template <typename T>
using sptr = std::shared_ptr<T>;

template <typename T>
std::shared_ptr<T> cuda_ptr_from_host(T *host_ptr, size_t size) {
  T *device_ptr;
  size_t totalBytes = size * sizeof(T);
  CUDA_CHECK_ERROR(cudaMalloc((void **)&device_ptr, totalBytes));
  CUDA_CHECK_ERROR(cudaMemcpy(device_ptr, host_ptr, totalBytes, cudaMemcpyHostToDevice));
  return std::shared_ptr<T>(device_ptr, [](T *ptr) { CUDA_CHECK_ERROR(cudaFree(ptr)); });
}

template <typename T>
std::shared_ptr<T> cuda_ptr_empty(size_t size) {
  T *device_ptr;
  size_t totalBytes = size * sizeof(T);
  CUDA_CHECK_ERROR(cudaMalloc((void **)&device_ptr, totalBytes));
  return std::shared_ptr<T>(device_ptr, [](T *ptr) { CUDA_CHECK_ERROR(cudaFree(ptr)); });
}

template <typename T>
std::shared_ptr<T> host_alloc(size_t size) {
  return std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
}

template <typename T>
void copy_to_host(T *host_ptr, T *device_ptr, size_t size) {
  CUDA_CHECK_ERROR(cudaMemcpy(host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void copy_to_host(T *host_ptr, std::shared_ptr<T> device_ptr, size_t size) {
  copy_to_host(host_ptr, device_ptr.get(), size);
}

template <typename T>
void copy_to_host(std::shared_ptr<T> host_ptr, T *device_ptr, size_t size) {
  copy_to_host(host_ptr.get(), device_ptr, size);
}

template <typename T>
void copy_to_host(std::shared_ptr<T> host_ptr, std::shared_ptr<T> device_ptr, size_t size) {
  copy_to_host(host_ptr.get(), device_ptr.get(), size);
}


constexpr size_t THREADS_PER_BLOCK = 256;