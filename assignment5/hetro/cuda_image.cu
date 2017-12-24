#include <stdio.h>
#include "image.h"

extern "C" {
#include "cuda_image.cuh"
}

static cudaEvent_t event_begin;
static cudaEvent_t event_before_kernel;
static cudaEvent_t event_after_kernel;
static cudaEvent_t event_end;

__global__ 
void grayscaleKernel(int pixels, int channels, byte *input, byte *output){
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < pixels) {
        byte r = input[i * channels];
        byte g = input[i * channels + 1];
        byte b = input[i * channels + 2];

        byte gray = (float)r * 0.299f + (float)g * 0.587f + (float)b * 0.114f;
        output[i] = gray;
    }
}


/* Writes the results to image_t output */
extern "C"
void cuda_grayscale(image_t *input, image_t *output) {
    // How many bytes is the image
    int pixels = input->w * input->h;
    int bytes = pixels * input->n;
    int threadBlockSize = 1024;
    int threadBlocks = ceil((float)pixels / (float)threadBlockSize);

    byte *d_in = NULL;
    byte *d_out = NULL;

    CUDA_CHECK(cudaEventCreate, &event_begin);
    CUDA_CHECK(cudaEventCreate, &event_before_kernel);
    CUDA_CHECK(cudaEventCreate, &event_after_kernel);
    CUDA_CHECK(cudaEventCreate, &event_end);

    CUDA_CHECK(cudaSetDevice, 0);
    CUDA_CHECK(cudaEventRecord, event_begin, CUDA_DEFAULT_STREAM);

    CUDA_CHECK(cudaMalloc, &d_in, bytes * sizeof(byte)); 
    CUDA_CHECK(cudaMalloc, &d_out, pixels * sizeof(byte));
    CUDA_CHECK(cudaMemcpyAsync, d_in, input->data, bytes * sizeof(byte), cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaEventRecord, event_before_kernel, CUDA_DEFAULT_STREAM);
    grayscaleKernel<<<threadBlocks, threadBlockSize, CUDA_DEFAULT_STREAM>>>(pixels, input->n, d_in, d_out);
    CUDA_CHECK(cudaEventRecord, event_after_kernel, CUDA_DEFAULT_STREAM);
    
    CUDA_CHECK(cudaGetLastError);
    
    CUDA_CHECK(cudaDeviceSynchronize);
    CUDA_CHECK(cudaMemcpyAsync, output->data, d_out, pixels * sizeof(byte), cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaFree, d_in);
    CUDA_CHECK(cudaFree, d_out);

    CUDA_CHECK(cudaEventRecord, event_end, CUDA_DEFAULT_STREAM);
}

__global__ 
void contrastKernel(int pixels, int mean, byte* data) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < pixels) { 
        if (data[i] > mean) {
            float d1 = (float)(data[i] - mean) / 255.0f;
            float d2 = 1.0f - ((float)mean / 255.0f);
            data[i] = (byte)((pow(d1, 0.5f) / pow(d2, 0.5f))* 255.0f);
        }
        else {
            data[i] = 0;
        }
    }
}

extern "C"
void cuda_contrast(image_t *image, int mean) {
    // Only use contrast on grayscale images
    if (image->n != 1) {
        return;
    }

    int pixels = image->w * image->h;
    int threadBlockSize = 1024;
    int threadBlocks = ceil((float)pixels / (float)threadBlockSize);
    
    byte *device = NULL;

    CUDA_CHECK(cudaEventCreate, &event_begin);
    CUDA_CHECK(cudaEventCreate, &event_before_kernel);
    CUDA_CHECK(cudaEventCreate, &event_after_kernel);
    CUDA_CHECK(cudaEventCreate, &event_end);

    CUDA_CHECK(cudaSetDevice, 0);
    CUDA_CHECK(cudaEventRecord, event_begin, CUDA_DEFAULT_STREAM);

    CUDA_CHECK(cudaMalloc, &device, pixels * sizeof(byte)); 
    CUDA_CHECK(cudaMemcpyAsync, device, image->data, pixels * sizeof(byte), cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaEventRecord, event_before_kernel, CUDA_DEFAULT_STREAM);
    contrastKernel<<<threadBlocks, threadBlockSize, CUDA_DEFAULT_STREAM>>>(pixels, mean, device);
    CUDA_CHECK(cudaEventRecord, event_after_kernel, CUDA_DEFAULT_STREAM);
    CUDA_CHECK(cudaGetLastError);
    
    CUDA_CHECK(cudaDeviceSynchronize);
    CUDA_CHECK(cudaMemcpy, image->data, device, pixels * sizeof(byte), cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaFree, device);

    CUDA_CHECK(cudaEventRecord, event_end, CUDA_DEFAULT_STREAM);
}

__device__
int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

__global__
void smoothingKernel(int pixels, int width, int height, byte *input, byte *output) {
    int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
    // The weights of the neighbourhood values, the sum is 81
    byte T[25] = {1, 2, 3, 2, 1, 2, 4, 6, 4, 2, 3, 6, 9, 6, 3, 2, 4, 6, 4, 2, 1, 2, 3, 2, 1};

    int image_pixels = width * height;
    if (thread_index < pixels) {
        unsigned int sum = 0;
        // Loop over the neighbourhood
        for (int i = 0; i < 25; i++) {
                int row = i / 5;
                int column = i % 5;
                int index = mod(thread_index + column - 2  + (row - 2) * width, image_pixels);

                sum += T[row * 5 + column] * input[index];
        }
        output[thread_index] = sum / 81;
    }
}

// Need to add extra image for boundary conditions
extern "C"
void cuda_smoothing(image_t *image, image_t *original) {
    int pixels = image->w * image->h;
    // Add some extra read only pixels to the bottom pixels
    int padding = image->w * 3;
    int threadBlockSize = 1024;
    int threadBlocks = ceil((float)pixels / (float)threadBlockSize);
    
    byte *d_in = NULL;
    byte *d_out = NULL;
    
    CUDA_CHECK(cudaEventCreate, &event_begin);
    CUDA_CHECK(cudaEventCreate, &event_before_kernel);
    CUDA_CHECK(cudaEventCreate, &event_after_kernel);
    CUDA_CHECK(cudaEventCreate, &event_end);

    CUDA_CHECK(cudaSetDevice, 0);
    CUDA_CHECK(cudaEventRecord, event_begin, CUDA_DEFAULT_STREAM);

    CUDA_CHECK(cudaMalloc, &d_in, (pixels + padding) * sizeof(byte));
    CUDA_CHECK(cudaMalloc, &d_out, pixels * sizeof(byte)); 
    CUDA_CHECK(cudaMemcpy, d_in, image->data, (pixels + padding) * sizeof(byte), cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaEventRecord, event_before_kernel, CUDA_DEFAULT_STREAM);
    smoothingKernel<<<threadBlocks, threadBlockSize, CUDA_DEFAULT_STREAM>>>(pixels, original->w, original->h, d_in, d_out);
    CUDA_CHECK(cudaEventRecord, event_after_kernel, CUDA_DEFAULT_STREAM);
    CUDA_CHECK(cudaGetLastError);
    
    CUDA_CHECK(cudaDeviceSynchronize);
    CUDA_CHECK(cudaMemcpy, image->data, d_out, pixels * sizeof(byte), cudaMemcpyDeviceToHost);

    CUDA_CHECK(cudaEventRecord, event_end, CUDA_DEFAULT_STREAM);

    CUDA_CHECK(cudaFree, d_in);
    CUDA_CHECK(cudaFree, d_out);

   
}