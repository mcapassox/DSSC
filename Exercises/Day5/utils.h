#include <stdio.h>
extern "C"
double* cuda_allocate(const size_t size);
void cuda_copy(double* dest, double* src, const size_t size, const size_t dir);
void cuda_free(double* matrix);
void populate(double*  matrix_in, const size_t size);
void print_matrix(const double* matrix_in, const size_t n);
