#include <stdio.h>
#include <math.h>
#include "utils.h"

#define N_SAMPLE 1E2
#define TILE_DIM 32

 // naive version of cuda transpose
__global__ void cuda_transpose_slow(const double *matrix_in, double *matrix_out, const size_t n) {
    
    // calculate grid index
    size_t x = threadIdx.x+blockIdx.x*blockDim.x;
    size_t y = threadIdx.y+blockIdx.y*blockDim.y;
    
    // calculate matrix index
    size_t in_index = x+y*n;
    size_t out_index = y+x*n;
    
    // transpose matrix
    if ((x < n) && (y < n)) {
        matrix_out[out_index] = matrix_in[in_index];
    }
}


// coalesced cuda transpose
__global__ void cuda_transpose_fast(const double *matrix_in, double *matrix_out, const size_t n) {
	
    // define tile
    __shared__ double tile[TILE_DIM][TILE_DIM];
	
    // calculate grid index
    size_t x = blockIdx.x*TILE_DIM + threadIdx.x;
    size_t y = blockIdx.y*TILE_DIM + threadIdx.y;
    
    // calculate matrix index
    size_t index_in = x + y*n;
	
    // calculate transposed grid index
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y; 
    
    // calculate transposed matrix index
    size_t index_out = x + y*n;
	
    // read row-maj, write row-maj
    tile[threadIdx.y][threadIdx.x] = matrix_in[index_in];

    __syncthreads();
   // read col-maj, write row-maj
   matrix_out[index_out] = tile[threadIdx.x][threadIdx.y];
   
}

// cpu transpose
void naive(const double* matrix_in, double* matrix_out, const size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix_out[j*n+i] = matrix_in[i*n+j];
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
    printf("%d,", TILE_DIM);
    
    // calculate matrix size in bytes
    const size_t size = n*n * sizeof(double);
    // Matrix to be transposed
    double *matrix_in;
    // Transposed matrix (naive)
    double *matrix_naive;
    // Transposed matrix (slow)
    double *matrix_slow, *dev_matrix_slow_in, *dev_matrix_slow_out;
    // Transposed matrix (coalesced)
    double *matrix_b, *dev_matrix_b_in, *dev_matrix_b_out;

    // Host allocation
    matrix_in = (double*)malloc(size);
    matrix_naive = (double*)malloc(size);
    matrix_slow = (double*)malloc(size);
    matrix_b = (double*)malloc(size);

    // CUDA allocation
    dev_matrix_slow_in = cuda_allocate(size);
    dev_matrix_slow_out = cuda_allocate(size);

    // fill matrix
    populate(matrix_in, n);


    // naively compute the transpose (CPU)
    cudaEventRecord(startEvent, 0);
    //naive(matrix_in, matrix_naive, n);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    //printf("TIMER_naive: %f\n", ms);

    // ............. slow NO coalesced .............

    // compute the size of the grid (for ex: 512 = 2^4*2^5)
    size_t square = n/sqrt(thread_per_block)+1;

    size_t power = log2((double)thread_per_block);
    size_t nX_threadPB = pow(2, (int)power/2);
    size_t nY_threadPB = thread_per_block/nX_threadPB;

    size_t nX_block = n/nX_threadPB;
    size_t nY_block = n/nY_threadPB;

    dim3 dimGrid(nX_block, nY_block, 1);
    dim3 dimBlock(nX_threadPB, nY_threadPB, 1);

    /*printf("GRID:\n x: %f\n y: %f\n", n/sqrt(thread_per_block), n/sqrt(thread_per_block));
    printf("BLOCK:\n x: %f\n y: %f\n", sqrt(thread_per_block), sqrt(thread_per_block));*/

    // copy input to device
    cuda_copy(dev_matrix_slow_in, matrix_in, size, 0);

    cudaEventRecord(startEvent, 0);

    // Take N_SAMPLES
    for (size_t i = 0; i < N_SAMPLE; i++) {
        cuda_transpose_slow<<< dimGrid, dimBlock >>>(dev_matrix_slow_in, dev_matrix_slow_out, n);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    // copy device result back to host
    cuda_copy(matrix_slow, dev_matrix_slow_out, size, 1);

    //printf("TIMER_slow: %f\n", ms/N_SAMPLE);
    printf("%f,", ms/N_SAMPLE);
    
    // free memory
    cuda_free(dev_matrix_slow_in);
    cuda_free(dev_matrix_slow_out);
    
    // Device allocation
    dev_matrix_b_in = cuda_allocate(size);
    dev_matrix_b_out = cuda_allocate(size);


    // ............. coalesced .............
    dim3 dimGrid1(n/TILE_DIM, n/TILE_DIM, 1);
    dim3 dimBlock1(TILE_DIM, TILE_DIM, 1);

    // copy inputs to device
    cuda_copy(dev_matrix_b_in, matrix_in, size, 0);

    cudaEventRecord(startEvent, 0);

    // Take n samples
    for (size_t i = 0; i < N_SAMPLE; i++) {
        cuda_transpose_fast<<<dimGrid1, dimBlock1>>>(dev_matrix_b_in, dev_matrix_b_out, n);
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);


    //printf("TIMER_coalesced: %f\n", ms/N_SAMPLE);
    printf("%f\n", ms/N_SAMPLE);
    
    // copy result back to host
    cuda_copy(matrix_b, dev_matrix_b_out, size, 1);
 
    // check results
    
    /*size_t control1 = 1;
    size_t control2 = 1;

    for (size_t i = 0; i < n*n; i++) {
        if (matrix_slow[i] != matrix_naive[i])
            control1 = 0;
        if (matrix_b[i] != matrix_naive[i])
            control2 = 0;
    }*/


    //printf("\nslow: %lu\ncoalesced: %lu\n", control1, control2);

    free(matrix_in);
    free(matrix_naive);
    free(matrix_slow);
    free(matrix_b);
    
    // free memory
    cuda_free(dev_matrix_b_in);
    cuda_free(dev_matrix_b_out);

    return 0;
}
