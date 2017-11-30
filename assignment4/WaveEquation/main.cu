#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "generatedata.h"
#include "file.h"
#include "simulate.h"


int main(int argc, char *argv[])
{
    float *old, *current, *next;
    int t_max, i_max, threadBlockSize;

    /* Parse commandline args: i_max t_max threadBlockSize */
    if (argc < 4) {
        printf("Usage: %s i_max t_max threadBlockSize [initial_data]\n", argv[0]);
        printf(" - i_max: number of discrete amplitude points, should be >2\n");
        printf(" - t_max: number of discrete timesteps, should be >=1\n");
        printf(" - threadBlockSize: the threadblocksize used for the simulation, "
                "should be >=1\n");
        printf(" - initial_data: select what data should be used for the first "
                "two generation.\n");
        printf("   Available options are:\n");
        printf("    * sin: one period of the sinus function at the start.\n");
        printf("    * sinfull: entire data is filled with the sinus.\n");
        printf("    * gauss: a single gauss-function at the start.\n");
        printf("    * file <2 filenames>: allows you to specify a file with on "
                "each line a float for both generations.\n");

        return EXIT_FAILURE;
    }

    i_max = atoi(argv[1]);
    t_max = atoi(argv[2]);
    threadBlockSize = atoi(argv[3]);

    if (i_max < 3) {
        printf("argument error: i_max should be >2.\n");
        return EXIT_FAILURE;
    }
    if (t_max < 1) {
        printf("argument error: t_max should be >=1.\n");
        return EXIT_FAILURE;
    }
    if (threadBlockSize < 8) {
        printf("argument error: threadBlockSize should be >=8.\n");
        return EXIT_FAILURE;
    }

    /* Allocate and initialize buffers. */
    old = (float *)malloc(i_max * sizeof(float));
    current = (float *)malloc(i_max * sizeof(float));
    next = (float *)malloc(i_max * sizeof(float));

    if (old == NULL || current == NULL || next == NULL) {
        fprintf(stderr, "Could not allocate enough memory, aborting.\n");
        return EXIT_FAILURE;
    }

    memset((void *)old, 0, i_max * sizeof(float));
    memset((void *)current, 0, i_max * sizeof(float));
    memset((void *)next, 0, i_max * sizeof(float));

    /* How should we will our first two generations? */
    if (argc > 4) {
        if (strcmp(argv[4], "sin") == 0) {
            fill(old, 1, i_max/4, 0, 2*3.14, sin);
            fill(current, 2, i_max/4, 0, 2*3.14, sin);
        } else if (strcmp(argv[4], "sinfull") == 0) {
            fill(old, 1, i_max-2, 0, 10*3.14, sin);
            fill(current, 2, i_max-3, 0, 10*3.14, sin);
        } else if (strcmp(argv[4], "gauss") == 0) {
            fill(old, 1, i_max/4, -3, 3, gauss);
            fill(current, 2, i_max/4, -3, 3, gauss);
        } else if (strcmp(argv[4], "file") == 0) {
            if (argc < 7) {
                printf("No files specified!\n");
                return EXIT_FAILURE;
            }
            file_read_double_array(argv[5], old, i_max);
            file_read_double_array(argv[6], current, i_max);
        } else {
            printf("Unknown initial mode: %s.\n", argv[4]);
            return EXIT_FAILURE;
        }
    } else {
        /* Default to sinus. */
        fill(old, 1, i_max/4, 0, 2*3.14, sin);
        fill(current, 2, i_max/4, 0, 2*3.14, sin);
    }

    /* Call the actual simulation that should be implemented in simulate.c. */
    simulate(i_max, t_max, threadBlockSize, old, current, next);
    file_write_double_array("result.txt", next, i_max);

    free(old);
    free(current);
    free(next);

    return EXIT_SUCCESS;
}
