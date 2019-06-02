#include<stdlib.h>
#include <stdio.h>

extern inline void vec_sum(int* out, const int* in, const size_t size)
{
    for(size_t i = 0; i<size; i++)
        out[i] += in[i];
}

void swapPP(int** a, int** b)
{
    int* t = *a;
    *a = *b;
    *b = t;
}

void fill_vec(int* v, const int x, const size_t size)
{
    for(size_t i = 0; i<size; i++)
        v[i] = x;
}

int **callocate_matrix(const size_t rows, const size_t cols)
{
   int **A=(int **)calloc(rows, sizeof(int *));
   for (size_t i=0; i<rows; i++) {
     A[i]=(int *)calloc(cols, sizeof(int));
   }
   return A;
}

void deallocate_matrix(int **A, const size_t rows)
{
  for (size_t i=0; i<rows; i++) {
      free(A[i]);
  }
  free(A);
}

void print_matrix(int** A, const size_t n_rows, const size_t n_cols){
  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_cols; j++){
    printf("%d ", A[i][j]);
    }
    printf("\n");
  }
}

void print_vector(int* A, size_t n_cols){
    for (size_t j = 0; j < n_cols; j++){
    printf("%d ", A[j]);
    }
    printf("\n");
  }

void set_one(int** A, const size_t n_rows, const size_t offset){
  for (size_t i = 0; i < n_rows; i++) {
    A[i][offset+i] = 1;
  }
}

void set_one_vect(int* A,  const size_t offset){
    A[offset] = 1;
}

void print_vectMat(const int* matrix, const size_t matrix_size)
{
    for(size_t i=0; i<matrix_size; i++)
    {
        for(size_t j=0; j<matrix_size; j++)
            printf("%d ", matrix[i*matrix_size+j]);
        printf("\n");
    }
}

void copy(int* a, int* b, const size_t size){
  for (size_t i = 0; i < size; i++) {
    b[i] = a[i];
  }
}
