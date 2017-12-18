#include <stdio.h>
#include "image.h"

extern "C" {
#include "cuda_image.cuh"
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ 
void grayscaleKernel(int pixels, int channels, byte *input, byte *output){
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < pixels) {
        byte r = input[i * channels];
        byte g = input[i * channels + 1];
        byte b = input[i * channels + 2];

        byte gray = (r + g + b) / 3;
        output[i] = gray;
    }
}


/* Writes the results to image_t output */
extern "C"
void rgb_to_grayscale(image_t *input, image_t *output) {
    // How many bytes is the image
    int pixels = input->w * input->h;
    int bytes = pixels * input->n;
    int threadBlockSize = 1024;
    int threadBlocks = ceil((float)pixels / (float)threadBlockSize);

    printf("Tread blocks: %i\n", threadBlocks);
    printf("Pixels: %i\n", pixels);
    printf("bytes: %i\n", bytes);
    printf("sizeof byte: %i\n", sizeof(byte));

    output->data = (byte*)malloc(sizeof(byte) * pixels);
    // Create output image with the same dimensions
    output->w = input->w;
    output->h = input->h;
    // The gray output has only one channel
    output->n = 1;

    byte *d_in = NULL;
    byte *d_out = NULL;

    gpuErrchk(cudaMalloc(&d_in, bytes * sizeof(byte))); 
    gpuErrchk(cudaMalloc(&d_out, pixels * sizeof(byte)));
    gpuErrchk(cudaMemcpy(d_in, input->data, bytes * sizeof(byte), cudaMemcpyHostToDevice));

    grayscaleKernel<<<threadBlocks, threadBlockSize>>>(pixels, input->n, d_in, d_out);

    // Wait for all kernels to finish
    gpuErrchk(cudaDeviceSynchronize());

    // Assuming output->data has enough memory allocated
    gpuErrchk(cudaMemcpy(output->data, d_out, pixels * sizeof(byte), cudaMemcpyDeviceToHost));

    // for (int i = 0; i < 100; i++) {
    //     printf("%i\n", output->data[i]);
    // }

    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
}
