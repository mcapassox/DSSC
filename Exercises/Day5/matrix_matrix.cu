#include <stdio.h>
#include <math.h>
#include "utils.h"

#define TILE_DIM 16
#define N_SAMPLE 10

// 1D version of matrix matrix multiplication (runs out of # of blocks)
__global__ void cuda_1D(const double *a, const double *b, double *matrix_out, const size_t n) {

    // j+i*n;
    size_t index = threadIdx.x+blockIdx.x*blockDim.x;
    size_t i = (index/n)*n;
    size_t j = index%n;
    double temp = 0;

    for (size_t k = 0; k < n; k++) {
        temp += a[i+k]*b[j+k*n];
    }

    matrix_out[index] = temp;

}

// 2D version CUDA MM multiplication by definition (row x col)
__global__ void cuda_MM(const double *a, const double *b, double *matrix_out, const size_t n) {
    
    // calculate grid index
    size_t x = threadIdx.x+blockIdx.x*blockDim.x;
    size_t y = threadIdx.y+blockIdx.y*blockDim.y;
    
    // calculate matrix index
    size_t index = x + y*n;

    double temp = 0;

    if (y < n && x < n) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < n; i++) {
            temp += a[y*n+i] * b[i*n+x];
        }
    }
    matrix_out[index] = temp;
}

// 2D version of CUDA MM multiplication with transposed b matrix (row x row)
__global__ void cuda_MMT(const double *a, const double *b, double *matrix_out, const size_t n) {
    
    // calculate grid index
    size_t x = threadIdx.x+blockIdx.x*blockDim.x;
    size_t y = threadIdx.y+blockIdx.y*blockDim.y;
    
    // calculate matrix index
    size_t index = x + y*n;

    double temp = 0;

    if (y < n && x < n) {
        for (int i = 0; i < n; i++) {
            temp += a[y*n+i] * b[x*n+i];
        }
    }
    matrix_out[index] = temp;
}

// CUDA transpose function
__global__ void cuda_transpose_fast(const double *matrix_in, double *matrix_out, const size_t n) {

    __shared__ double tile[TILE_DIM][TILE_DIM];

    size_t x = blockIdx.x*TILE_DIM + threadIdx.x;
    size_t y = blockIdx.y*TILE_DIM + threadIdx.y;

    // j+ i*n
    size_t index_in = x + y*n;

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    size_t index_out = x + y*n;

    tile[threadIdx.y][threadIdx.x] = matrix_in[index_in];

    __syncthreads();

    matrix_out[index_out] = tile[threadIdx.x][threadIdx.y];

}

// CPU matrix multiplication
void naive(const double* a, const double* b, double* matrix_out, const size_t n) {
    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                matrix_out[i*n+k] += a[i*n+j]*b[j*n+k];
            }
        }
    }
}

// CPU  version of  MM multiplication with transposed b matrix (row x row)
void naiveT(const double* a, const double* b, double* matrix_out, const size_t n) {
    for (size_t k = 0; k < n; k++) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                matrix_out[i*n+k] += a[i*n+j]*b[j+k*n];
            }
        }
    }
}

int main(int argc, char *argv[]) {

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    float ms;

    // create events for timing
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    if(argc < 3){
      printf("Two arguments need to be passed: n, thread_per_block. RETURNING. \n");
      return -1;
    }

    const size_t n = atoi(argv[1]);
    const size_t thread_per_block = atoi(argv[2]);

    printf("%lu,", thread_per_block);
    
    // calculate size in bytes
    const size_t size = n*n*sizeof(double);
    
    // Matrices to be multiplied
    double *a, *b;
    // result matrix (naive)
    double *matrix_naive, *matrix_naiveT;
    // result matrices CUDA
    double *matrix_1D, *matrix_MM, *matrix_MMT, *bT;
    // device matrices
    double *dev_matrix_a, *dev_matrix_b, *dev_matrix_bT, *dev_matrix_c, *dev_matrix_cT;

    // Host allocation
    a = (double*)malloc(size);
    b = (double*)malloc(size);
    bT = (double*)malloc(size);

    matrix_naive = (double*)calloc(n*n, sizeof(double));
    matrix_naiveT = (double*)calloc(n*n, sizeof(double));
    matrix_1D = (double*)calloc(n*n, sizeof(double));
    matrix_MM = (double*)calloc(n*n, sizeof(double));
    matrix_MMT = (double*)calloc(n*n, sizeof(double));

    // Device allocation
    dev_matrix_a = cuda_allocate(size);
    dev_matrix_b = cuda_allocate(size);
    dev_matrix_c = cuda_allocate(size);

    // populate the matrices
    populate(a, n);
    populate(b, n);


    // naively compute the moltiplication (CPU)
    cudaEventRecord(startEvent, 0);
    //naive(a, b, matrix_naive, n);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    //printf("TIMER_CPU_naive: %f\n", ms);

    // ............. 1D .............

    /*printf("GRID:\n x: %f\n y: %f\n", n/sqrt(thread_per_block), n/sqrt(thread_per_block));
    printf("BLOCK:\n x: %f\n y: %f\n", sqrt(thread_per_block), sqrt(thread_per_block));*/

    cuda_copy(dev_matrix_a, a, size, 0);
    cuda_copy(dev_matrix_b, b, size, 0);

    cudaEventRecord(startEvent, 0);

    // Take N_SAMPLEs
    for (size_t i = 0; i < N_SAMPLE; i++) {
        cuda_1D<<< n*n/thread_per_block, thread_per_block>>>(dev_matrix_a, dev_matrix_b, dev_matrix_c, n);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    //printf("TIMER_1D: %f\n", ms/N_SAMPLE);
    printf("%f,", ms/N_SAMPLE);
    // copy device result back to host
    cuda_copy(matrix_1D, dev_matrix_c, size, 1);

    // ............. MM .............
    size_t square = n/sqrt(thread_per_block)+1;

    size_t power = log2((double)thread_per_block);
    size_t nX_threadPB = pow(2, (int)power/2);
    size_t nY_threadPB = thread_per_block/nX_threadPB;

    size_t nX_block = n/nX_threadPB;
    size_t nY_block = n/nY_threadPB;

    dim3 dimGrid(nX_block, nY_block, 1);
    dim3 dimBlock(nX_threadPB, nY_threadPB, 1);

    /*printf("GRID:\n x: %lu\n y: %lu\n", nX_block, nY_block);
    printf("BLOCK:\n x: %lu\n y: %lu\n", nX_threadPB, nY_threadPB);*/

    cudaEventRecord(startEvent, 0);
    
    // Take N_SAMPLEs
    for (size_t i = 0; i < N_SAMPLE; i++) {
        cuda_MM<<< dimGrid, dimBlock >>>(dev_matrix_a, dev_matrix_b, dev_matrix_c, n);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    //printf("TIMER_MM: %f\n", ms/N_SAMPLE);
    printf("%f,", ms/N_SAMPLE);


    // copy device result back to host
    cuda_copy(matrix_MM, dev_matrix_c, size, 1);

    cuda_free(dev_matrix_c);

    // .............TRANSPOSE MATRIX b (FOR CUDA MMT)  .............

    /*printf("GRID:\n x: %f\n y: %f\n", n/sqrt(thread_per_block), n/sqrt(thread_per_block));
    printf("BLOCK:\n x: %f\n y: %f\n", sqrt(thread_per_block), sqrt(thread_per_block));*/


    // ............. COALESCING .............
    dim3 dimGrid1(n/TILE_DIM, n/TILE_DIM, 1);
    dim3 dimBlock1(TILE_DIM, TILE_DIM, 1);

    // Allocate space for the transpose on device
    dev_matrix_bT = cuda_allocate(size);

    cudaEventRecord(startEvent, 0);

    // Take n samples
    for (size_t i = 0; i < N_SAMPLE; i++) {
        cuda_transpose_fast<<<dimGrid1, dimBlock1>>>(dev_matrix_b, dev_matrix_bT, n);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);


    //printf("TIMER_trans: %f\n", ms/N_SAMPLE);
    printf("%f,", ms/N_SAMPLE);

    cuda_free(dev_matrix_b);
    dev_matrix_cT = cuda_allocate(size);

    dim3 dimGrid2(nX_block, nY_block, 1);
    dim3 dimBlock2(nX_threadPB, nY_threadPB, 1);
    cudaEventRecord(startEvent, 0);
    
    // Take N_SAMPLEs
    for (size_t i = 0; i < N_SAMPLE; i++) {
        cuda_MMT<<< dimGrid2, dimBlock2 >>>(dev_matrix_a, dev_matrix_bT, dev_matrix_cT, n);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    //printf("TIMER_MMT: %f\n", ms/N_SAMPLE);
    printf("%f", ms/N_SAMPLE);

    // copy device result back to host
    cuda_copy(matrix_MMT, dev_matrix_cT, size, 1);

    cudaEventRecord(startEvent, 0);
    
    // Take N_SAMPLEs
    for (size_t i = 0; i < N_SAMPLE; i++) {
        //naiveT(a, bT, matrix_naiveT, n);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    //printf("TIMER_CPU_NAIVET: %f\n", ms/N_SAMPLE);

    /*size_t control1 = 1;
    size_t control2 = 1;
    size_t control3 = 1;

    for (size_t i = 0; i < n*n; i++) {
        if (matrix_1D[i] != matrix_naive[i])
            control1 = 0;
        if (matrix_MMT[i] != matrix_MM[i])
            control2 = 0;
        if (matrix_naiveT[i] != matrix_MM[i])
            control3 = 0;
    }*/


    //printf("\nCUDA: %lu\nCUDAT: %lu\nCPU_NAIVET: %lu\n\n", control1, control2, control3);

    free(a);
    free(b);
    free(bT);

    free(matrix_naive);
    free(matrix_naiveT);
    free(matrix_1D);
    free(matrix_MM);
    free(matrix_MMT);

    cuda_free(dev_matrix_a);
    cuda_free(dev_matrix_bT);
    cuda_free(dev_matrix_cT);


    return 0;
}
