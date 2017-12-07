#include <stdio.h>
#include <stdlib.h>
#include "simulate.h"
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>

// TODO:
// Different mode support
// different threadblocksize support
// working modes

using namespace std;

/* Modes:
 * 0: Checking for out of bounds  
 */
int MODE = 0;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


int createDevices(float *deviceA, float *deviceB, float *deviceResult, int size) {
    
    checkCudaCall(cudaMalloc((void **) &deviceA, size));
    if (deviceA == NULL) {
        cout << "could not allocate memory!" << endl;
        return 0;
    }

    
    checkCudaCall(cudaMalloc((void **) &deviceB, size));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        cout << "could not allocate memory!" << endl;
        return 0;
    }

    
    checkCudaCall(cudaMalloc((void **) &deviceResult, size));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        cout << "could not allocate memory!" << endl;
        return 0;
    }
    return 1;
}

__global__ void simulateBoundCheckKernel(float* old, float* current, float* next, int i_max) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (index > 0 && index < i_max){
            next[index] = 2.0 * current[index] - old[index] + 0.15 * 
                (current[index - 1] - (2.0 * current[index] - current[index + 1]));
        }          
}

__global__ void simulateNoBoundCheckKernel(float* old, float* current, float* next) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
        next[index] = 2.0 * current[index] - old[index] + 0.15 * 
            (current[index - 1] - (2.0 * current[index] - current[index + 1]));

}

void simulateNoBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize) {
    int padding = 0;

    
    if ( n % threadBlockSize != 0) {
        padding = threadBlockSize - (n % threadBlockSize);
    }
    padding += 2; // padding because otherwise segmentation errors
    

    int size = n * sizeof(float) + padding * sizeof(float); 

    fprintf(stderr, "Size: %i\n", size);
    fprintf(stderr, "Padding: %i\n", padding);

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    float* deviceB = NULL;
    float* deviceResult = NULL;
    
    // Exit if there is an error during allocation
    if (!createDevices(deviceA, deviceB, deviceResult, size)) {
        return;
    } 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaCall(cudaMemcpy(deviceA + 1, a, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB + 1, b, n*sizeof(float), cudaMemcpyHostToDevice));
    
 
    fprintf(stderr, "Test before record\n");
    cudaEventRecord(start, 0);
    fprintf(stderr, "Test after record\n");
    fprintf(stderr, "DeviceA minus one: %p\n", deviceA - 1);
    *(deviceA) = 0.0f;
    *(deviceB) = 0.0f;
    *(deviceResult) = 0.0f;
    fprintf(stderr, "Test after devices\n");

    deviceA++;
    deviceB++;
    deviceResult++;

    for (int t = 0; t < t_max; t++) {
        fprintf(stderr, "Test before kernel\n");
        simulateNoBoundCheckKernel<<<ceil(n/(float)threadBlockSize), threadBlockSize>>>(deviceA, deviceB, deviceResult);
        deviceResult[n - 2] = 0.0;
        deviceResult[1] = 0.0;

        float *temp = deviceA;
        deviceA = deviceB;
        deviceB = deviceResult;
        deviceResult = temp;

        fprintf(stderr, "In cycle: %i\n", t);
    }
    deviceA--;
    deviceB--;
    deviceResult--;
    cudaEventRecord(stop, 0);
     
    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());
    
    checkCudaCall(cudaMemcpy(result, deviceB + 1, n * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    // print the time the kernel invocation took, without the copies!
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "kernel invocation took " << elapsedTime << " milliseconds" << endl;

}

void simulateBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize) {
    int size = n * sizeof(float); 

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    float* deviceB = NULL;
    float* deviceResult = NULL;
    
    // Exit if there is an error during allocation
    if (!createDevices(deviceA, deviceB, deviceResult, size)) {
        return;
    } 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(stderr, "before memcpy %p\n", deviceA);
    checkCudaCall(cudaMemcpy(deviceA, a, size, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, size, cudaMemcpyHostToDevice));
    fprintf(stderr, "after memcpy\n");

    cudaEventRecord(start, 0);
    for (int i = 0; i < t_max; i++) {
        simulateBoundCheckKernel<<<ceil(n/(float)threadBlockSize), threadBlockSize>>>(deviceA, deviceB, deviceResult, n);

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


float *simulate(const int i_max, const int t_max, const int threadBlockSize,
                float *old_array, float *current_array, float *next_array) {
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
                     
    return next_array;
}