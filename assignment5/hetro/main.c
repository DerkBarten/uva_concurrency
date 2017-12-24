#include <stdio.h>
#include "timer.h"
#include "image.h"
#include "hetro_image.h"
#include <omp.h>


int main(int argc, char *argv[]) {
    if (argc > 2) {
        // Create two image objects on the stack
        image_t input;
        image_t output;

        omp_set_dynamic(0);
        omp_set_num_threads(8);

        load_image(argv[1], &input);
        timer_start();
        hetro_image(&input, &output);
        double t = timer_end();
        fprintf(stderr, "time hetro: %f\n", t);

        save_image(argv[2], &output);

        // Free the data array in the image object
        unload_image(&input);
        unload_image(&output);
    }
    return 0;
}