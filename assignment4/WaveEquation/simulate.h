/*
 * simulate.h
 */

void simulate(const int i_max, const int t_max, const int threadBlockSize,
                float *old_array, float *current_array, float *next_array);

int createDevices(float *deviceA, float *deviceB, float *deviceResult, int size);

void simulateNoBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize);

void simulateBoundCheck(int n, float* a, float* b, float* result, int t_max, int threadBlockSize);