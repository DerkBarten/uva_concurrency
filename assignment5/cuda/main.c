#include <png.h>
#include "timer.h"
#include "image.h"
#include "cuda_image.cuh"

int main(int argc, char *argv[]) {
    if (argc > 2) {
        // Create two image objects on the stack
        image_t input;
        image_t output;

        load_image(argv[1], &input);
        timer_start();
        cuda_image(&input, &output);
        double t = timer_end();
        fprintf(stderr, "time cuda: %f\n", t);

        save_image(argv[2], &output);

        // Free the data array in the image object
        unload_image(&input);
        unload_image(&output);
    }
    return 0;
}