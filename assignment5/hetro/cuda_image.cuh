#include "image.h"

#define CUDA_DEFAULT_STREAM (0)

#ifdef __CUDACC__
#define CUDA_CHECK(f, ...) \
    check_cuda_error_code(f(__VA_ARGS__), #f, __FILE__, __LINE__)

static void check_cuda_error_code(cudaError_t error, const char *fun, const char *file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "error: %s:%d: %s: %s\n", file, line, fun, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

#endif

void cuda_grayscale(image_t *input, image_t *output);

void cuda_contrast(image_t *image, int mean);

void cuda_smoothing(image_t *image, image_t *original);