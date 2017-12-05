#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>

static void checkCudaCall(cudaError_t result);

__global__ void vectorSimulateKernel(float* previous, float* current, float* next, int i_max);

void vectorSimulateCuda(int n, float* a, float* b, float* result, int t_max);