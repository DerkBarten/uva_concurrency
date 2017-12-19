#include "image.h"

int main(int argc, char *argv[]) {
    if (argc > 2) {
        // Create two image objects on the stack
        image_t input;
        image_t output;

        load_image(argv[1], &input);
        openMP_grayscale(&input, &output);
        openMP_contrast(&output);
        openMP_smoothing(&output); 
        save_image(argv[2], &output);

        // Free the data array in the image object
        unload_image(&input);
        unload_image(&output);
    }
    return 0;
}