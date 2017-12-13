#include <png.h>
#include "image.h"

int main(int argc, char *argv[]) {
    if (argc > 2) {
        // Create two image objects on the stack
        image_t input;
        image_t output;

        load_image(argv[1], &input);
        rgb_to_grayscale(&input, &output);
        save_image(argv[2], &output);

        // Free the data array in the image object
        unload_image(&input);
        unload_image(&output);
    }
    return 0;
}