#include "image.h"
#include "timer.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
    if (argc > 2) {
        // Create two image objects on the stack
        image_t input;
        image_t output;

        load_image(argv[1], &input);

        timer_start();
        grayscale(&input, &output);
        contrast(&output);
        smoothing(&output);

        double t = timer_end();
        fprintf(stderr, "time sequential: %f\n", t);

        save_image(argv[2], &output);

        // Free the data array in the image object
        unload_image(&input);
        unload_image(&output);
    }
    return 0;
}