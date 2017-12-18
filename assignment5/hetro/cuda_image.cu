#include <stdio.h>
#include "image.h"

extern "C" {
#include "cuda_image.cuh"
}

__global__ 
void grayscaleKernel(image_t *input, image_t *output){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = i / input->w;
    i = i % input->w;

    if (j < input->h) {
        byte r = input->data[(i * input->w + j) * input->n];
        byte g = input->data[(i * input->w + j) * input->n + 1];
        byte b = input->data[(i * input->w + j) * input->n + 2];

        byte gray = (r + g + b) / 3;
        // Set the corresponding output pixel to the grayscale vaue
        output->data[i * input->w + j] = gray;
    }
}


/* Writes the results to image_t output */
extern "C"
void rgb_to_grayscale(image_t *input, image_t *output) {
    // How many bytes is the image
    int N = input->w * input->h * input->n;
    int threadBlockSize = 512;

    // Create output image with the same dimensions
    *output = *input; 
    output->data = (unsigned char*)malloc(sizeof(unsigned char) * input->w * input->h);
    // The gray output has only one channel
    output->n = 1;

    byte *d_in;
    byte *d_out;

    cudaMalloc((void**)&d_in, N * sizeof(byte)); 
    cudaMalloc((void**)&d_out, N * sizeof(byte));

    cudaMemcpy(d_in, input->data, N*sizeof(byte), cudaMemcpyHostToDevice);

    // Create images with pointers to the device buffers
    image_t d_img_in = *input;
    d_img_in.data = d_in;
    image_t d_img_out = *output;
    d_img_out.data = d_out;

    grayscaleKernel<<<(input->w * input->h) / threadBlockSize, threadBlockSize>>>(&d_img_in, &d_img_out);
    // Assuming output->data has enough memory allocated
    cudaMemcpy(output->data, d_img_out.data, N * sizeof(byte), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
