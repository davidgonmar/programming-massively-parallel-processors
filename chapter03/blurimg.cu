#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "cuda.h"
#include "../common/stb_image_write.h"
#include "../common/utils.cuh"
#include <iostream>

#define BLUR_SIZE 4
#define CHANNELS 3

__global__ void blurImgKernel(unsigned char *in, unsigned char *out,
                              unsigned int height, unsigned int width) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int totalR = 0;
  int totalG = 0;
  int totalB = 0;
  for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE; blurRow++) {
    for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE; blurCol++) {
      int startX = x + blurRow;
      int startY = y + blurCol;
      if (startX >= 0 && startX < width && startY >= 0 && startY < height) {
        int currentIdx = (startY * width + startX) * CHANNELS;
        totalR += in[currentIdx];
        totalG += in[currentIdx + 1];
        totalB += in[currentIdx + 2];
      }
    }
  }
  int outIdx = (y * width + x) * CHANNELS;
  // Multiply by 4 because the loop runs from -BLUR_SIZE to BLUR_SIZE-1, which
  // is BLUR_SIZE*2 on each axis
  out[outIdx] =
      static_cast<unsigned char>(totalR / (BLUR_SIZE * BLUR_SIZE * 4));
  out[outIdx + 1] =
      static_cast<unsigned char>(totalG / (BLUR_SIZE * BLUR_SIZE * 4));
  out[outIdx + 2] =
      static_cast<unsigned char>(totalB / (BLUR_SIZE * BLUR_SIZE * 4));
}

void blurImg(unsigned char *in, unsigned char *out, unsigned int height,
             unsigned int width) {
  sptr<unsigned char> d_in = cuda_ptr_from_host(in, height * width * CHANNELS);
  sptr<unsigned char> d_out = cuda_ptr_empty<unsigned char>(height * width * CHANNELS);

  dim3 threads(16, 16, 1);
  dim3 blocks(ceil(width / threads.x), ceil(height / threads.y), 1);

  blurImgKernel<<<blocks, threads>>>(d_in.get(), d_out.get(), height, width);

  copy_to_host(out, d_out, height * width * CHANNELS);
}

int main(int argc, char **argv) {
  // read an image
  int width, height, nChannels;

  // read from args the input and output file
  if (argc != 3) {
    std::cout << "Usage: ./blurimg <input_file> <output_file>" << std::endl;
    return -1;
  }

  unsigned char *data = stbi_load(argv[1], &width, &height, &nChannels, 0);

  if (data == NULL) {
    std::cout << "Failed to load image" << std::endl;
    return -1;
  }

  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;
  std::cout << "nChannels: " << nChannels << std::endl;

  unsigned char *out = new unsigned char[width * height * CHANNELS];

  std::cout << "Converting to greyscale" << std::endl;
  blurImg(data, out, height, width);

  stbi_write_jpg(argv[2], width, height, CHANNELS, out, 100);
}