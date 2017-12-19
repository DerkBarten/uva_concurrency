typedef unsigned char byte;

typedef struct {
    // Width, height, channels
    int w, h, n;
    // Pointer to the image data
    byte *data;
} image_t;

int load_image(char *filename, image_t *image);

int save_image(char *filename, image_t *image);

void unload_image(image_t *image);

int openMP_grayscale(image_t *input, image_t *output);

int openMP_contrast(image_t *input);

int openMP_smoothing(image_t *image);