/*
 * simulate.h
 */

float *simulate(const int i_max, const int t_max, const int threadBlockSize,
                float *old_array, float *current_array, float *next_array);

int createDevices(float *deviceA, float *deviceB, float *deviceResult, int size);

__global__ void simulateBoundCheckKernel(float* old, float* current, float* next, int i_max);

__global__ void simulateNoBoundCheckKernel(float* old, float* current, float* next);

void simulateNoBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize);

void simulateBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize);