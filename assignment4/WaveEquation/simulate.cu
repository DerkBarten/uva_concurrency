#include <stdio.h>
#include <stdlib.h>
#include "simulate.h"
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>

using namespace std;

/* Modes:
 * 0: Checking for array bounds in the kernel
 * 1: No checking for array bounds in the kernel (removes the if statement)
 */
int MODE = 0;

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


int createDevices(void  **deviceA, void**deviceB, void**deviceResult, int size) {
    
    checkCudaCall(cudaMalloc(deviceA, size));
    if (deviceA == NULL) {
        cout << "could not allocate memory!" << endl;
        return 0;
    }

    
    checkCudaCall(cudaMalloc(deviceB, size));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        cout << "could not allocate memory!" << endl;
        return 0;
    }

    
    checkCudaCall(cudaMalloc(deviceResult, size));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        cout << "could not allocate memory!" << endl;
        return 0;
    }
    return 1;
}


/* The simulation kernel without array bound checking */
__global__ void simulateNoBoundCheckKernel(float* old, float* current, float* next) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    next[index] = 2.0 * current[index] - old[index] + 0.15 * 
        (current[index - 1] - (2.0 * current[index] - current[index + 1]));

}

/* 
 * This function performs the wave equation simulation without the need to
 * check the bounds of the arrays in the kernel. 
 */
void simulateNoBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize) {
    // Calculate the padding size to make the datapoint array a multiple of the
    // thread block size
    int padding = 0;
    if ( n % threadBlockSize != 0) {
        padding = threadBlockSize - (n % threadBlockSize);
        // A float of extra padding to make sure the last thread does not access
        // memory that is not allocated
        padding += 1; 
    }
    int size = n * sizeof(float) + padding * sizeof(float); 

    fprintf(stderr, "Size: %i\n", size);
    fprintf(stderr, "Padding: %i\n", padding);

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    float* deviceB = NULL;
    float* deviceResult = NULL;
    
     // Exit if there is an error during allocation
     if (!createDevices((void **)&deviceA, (void **)&deviceB, (void **)&deviceResult, size)) {
        fprintf(stderr, "Error creating device\n");
        return;
    } 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy the datapoint arrays to the gpu
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(float), cudaMemcpyHostToDevice));
 
    cudaEventRecord(start, 0);

    // Increase the arrays on the gpu by one location because we do not want
    // the first index to access memory that is not allocated.
    deviceA++;
    deviceB++;
    deviceResult++;

    float zero = 0.0;
    for (int t = 0; t < t_max; t++) {
        simulateNoBoundCheckKernel<<<ceil(n/(float)threadBlockSize), threadBlockSize>>>(deviceA, deviceB, deviceResult);

        // Overwrite the right most datapoint with a zero to keep the random data
        // in the padding from interfering with the actual datapoints
        fprintf(stderr, "Test before zero cpy\n");
        checkCudaCall(cudaMemcpy(deviceResult + n - 1, &zero, sizeof(float), cudaMemcpyHostToDevice));
        fprintf(stderr, "Test after zero cpy\n");
        // The program throws an error on the memory copy, most likely due to
        // unaligned memory access.

        float *temp = deviceA;
        deviceA = deviceB;
        deviceB = deviceResult;
        deviceResult = temp;
    }
    deviceA--;
    deviceB--;
    deviceResult--;
    cudaEventRecord(stop, 0);
     
    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());
    
    // n - 1 because we ignored the first float when we copied to the device
    checkCudaCall(cudaMemcpy(result, deviceB, (n - 1) * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    // print the time the kernel invocation took, without the copies!
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "kernel invocation took " << elapsedTime << " milliseconds" << endl;

}

/* The simulation kernel with array bound checking */
__global__ void simulateBoundCheckKernel(float* old, float* current, float* next, int i_max) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    // The wave equation with 
    if (index > 0 && index < i_max - 1){
        next[index] = 2.0 * current[index] - old[index] + 0.15 * 
            (current[index - 1] - (2.0 * current[index] - current[index + 1]));
    }          
}

/* 
 * This function performs the wave equation simulation and needs to check the
 * bounds of the arrays in the kernel. 
 */
void simulateBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize) {
    int size = n * sizeof(float); 

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    float* deviceB = NULL;
    float* deviceResult = NULL;
    
    // Exit if there is an error during allocation
    if (!createDevices((void **)&deviceA, (void **)&deviceB, (void **)&deviceResult, size)) {
        fprintf(stderr, "Error creating device\n");
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaCall(cudaMemcpy(deviceA, a, size, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, size, cudaMemcpyHostToDevice));
    
    cudaEventRecord(start, 0);
    int blocks = ceil((float)n/(float)threadBlockSize);

    for (int t = 0; t < t_max; t++) {
        simulateBoundCheckKernel<<<blocks, threadBlockSize>>>(deviceA, deviceB, deviceResult, n);
        // Make sure the kernels are finished before swapping the buffers
        checkCudaCall(cudaDeviceSynchronize());

        float *temp = deviceA;
        deviceA = deviceB;
        deviceB = deviceResult;
        deviceResult = temp;
    }
    cudaEventRecord(stop, 0);
    
    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    checkCudaCall(cudaMemcpy(result, deviceB, size, cudaMemcpyDeviceToHost));
    
    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    // print the time the kernel invocation took, without the copies!
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "kernel invocation took " << elapsedTime << " milliseconds" << endl;
}


void simulate(const int i_max, const int t_max, const int threadBlockSize,
                float *old_array, float *current_array, float *next_array) {
    // Call the appropriate simulation function for the mode
    switch(MODE) {
        case 0:
            simulateBoundCheck(i_max, old_array, current_array, next_array, t_max, threadBlockSize);
            break;
        case 1:
            simulateNoBoundCheck(i_max, old_array, current_array, next_array, t_max, threadBlockSize);
            break;
        default:
            fprintf(stderr, "ERROR: No valid mode specified");
            break; 
    }
}