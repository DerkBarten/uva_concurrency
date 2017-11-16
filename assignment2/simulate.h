/*
 * simulate.h
 */

#pragma once
double wave(int i);

void buffer_swap(void);

double *simulate(const int i_max, const int t_max, double *old_array,
        double *current_array, double *next_array);
