#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "cuda.h"
#include "stb_image_write.h"
#include <iostream>

#define CHANNELS 3

__global__ void colorToGreyscaleKernel(unsigned char *in, unsigned char *out,
                                       unsigned int height,
                                       unsigned int width) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int dataStart = CHANNELS * (y * width + x);
  unsigned char r = in[dataStart];
  unsigned char g = in[dataStart + 1];
  unsigned char b = in[dataStart + 2];

  int dataOutStart = y * width + x;

  out[dataOutStart] = 0.21f * r + 0.71f * g + 0.07f * b;
}

void colorToGreyscale(unsigned char *in, unsigned char *out,
                      unsigned int height, unsigned int width) {
  unsigned char *d_in, *d_out;
  cudaMalloc(&d_in, sizeof(unsigned char) * height * width * CHANNELS);
  cudaMalloc(&d_out, sizeof(unsigned char) * height * width);

  cudaMemcpy(d_in, in, sizeof(unsigned char) * height * width * CHANNELS,
             cudaMemcpyHostToDevice);

  dim3 threads(16, 16, 1);
  dim3 blocks(ceil(width / threads.x), ceil(height / threads.y), 1);

  colorToGreyscaleKernel<<<blocks, threads>>>(d_in, d_out, height, width);

  std::cout << "printing out" << std::endl;

  cudaMemcpy(out, d_out, sizeof(unsigned char) * height * width,
             cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv) {
  // read an image
  int width, height, nChannels;

  unsigned char *data = stbi_load("dog.jpg", &width, &height, &nChannels, 0);

  if (data == NULL) {
    std::cout << "Failed to load image" << std::endl;
    return -1;
  }
  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;
  std::cout << "nChannels: " << nChannels << std::endl;

  unsigned char *out = new unsigned char[width * height * CHANNELS];

  std::cout << "Converting to greyscale" << std::endl;
  colorToGreyscale(data, out, height, width);

  std::cout << "Saving image. Printing raw data:" << std::endl;

  stbi_write_jpg("doggrayscale.jpg", width, height, 1, out, 100);
}