#include <stdio.h>

extern "C"

// CUDA allocating function, with naive error handling
double* cuda_allocate(const size_t size) {

    double *dev_matrix;

    if (cudaMalloc((void**)&dev_matrix, size) != cudaSuccess) {
        printf("Error allocating\n");
        return 0;
    }
    else {
        return dev_matrix;
    }

}

// cuda copy with naive error handling (0 = host to device, 1 = device to host)
 void cuda_copy(double* dest, double* src, const size_t size, const size_t dir) {

    // copy inputs to/from device (0 is from device to host)
    if (dir == 0) {
      if (cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice) != cudaSuccess) {
          printf("NOCOPY!\n\n");
      }
    }
    else{
      if (cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
          printf("NOCOPY!\n\n");
      }
    }
}

// CUDA free with naive error handling
 void cuda_free(double* matrix){
  if (cudaFree(matrix) != cudaSuccess) {
      printf("NOFREE!\n\n");
  }
}

// populate function
void populate(double*  matrix_in, const size_t size){
  for (size_t i = 0; i < size*size; i++) {
    matrix_in[i] = i;
  }
}

void print_matrix(const double* matrix_in, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%0.f ", matrix_in[i*n+j]);
        }
        printf("\n");
    }
}
