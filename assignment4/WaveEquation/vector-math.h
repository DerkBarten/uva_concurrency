#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>

static void checkCudaCall(cudaError_t result);

__global__ void vectorAddKernel(float* deviceA, float* deviceB, float* deviceResult);

void vectorAddCuda(int n, float* a, float* b, float* result);
