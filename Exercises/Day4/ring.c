#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include "utils.h"
#include "cptimer.h"

// single element being passed
void blocking_single(const int rank, const int size) {

  int X = rank, X2, sum = rank;
  double t1, t2;

  if(rank == 0)
  	t1 = seconds();

  // Each process receives from the previous process and sends to the next one
  for(size_t i=0; i<size-1; i++)
  {
      MPI_Send(&X, 1, MPI_INT, (rank+1)%size,    101, MPI_COMM_WORLD);
      MPI_Recv(&X2, 1, MPI_INT, (rank-1+size)%size, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      sum += X2;
      X = X2;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
  	t2 = seconds();

  printf("Block, single: Process: %d and the sum is: %d\n",rank,sum);

  if(rank==0)
      printf("Blocking Single element: elapsed-time: %f\n\n", t2-t1);
}

// vector being passed
void blocking_vec(const int rank, const int size, const int N) {

    double t1, t2;

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    	t1 = seconds();

    int* Xs = malloc(N*sizeof(int));
    int* X2s = malloc(N*sizeof(int));
    int* sums = malloc(N*sizeof(int));

    // fill vectors
    fill_vec(Xs, rank, N);
    fill_vec(sums, rank, N);

    // each process sends to the next one and receives from the previous one
    for(size_t i=0; i<size-1; i++)
    {
        MPI_Send(Xs, 1, MPI_INT, (rank+1)%size,    101, MPI_COMM_WORLD);
        MPI_Recv(X2s, 1, MPI_INT, (rank-1+size)%size, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // sum the vectors
        vec_sum(sums,X2s,N);

        // Swapping is only necessary to properly free memory
        swapPP(&Xs, &X2s);
    }

    // free memory
    free(Xs);
    free(X2s);
    free(sums);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    	t2 = seconds();

   printf("Vec: block: Process: %d and the sum is: %d\n",rank,sums[0]);

    if(rank==0)
      printf("Blocking Vector: elapsed-time: %f\n\n", t2-t1);
}

// single element being passed, non blocking
void nonblocking_single(const int rank, const int size) {

  double t1, t2;

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
  	t1 = seconds();

  MPI_Request request;
  MPI_Status status;

  int X = rank, X2, sum = rank;

  // each process sends to the next one and receives from the previous one
  for(size_t i=0; i<size-1; i++)
  {
      MPI_Isend(&X, 1, MPI_INT, (rank+1)%size,101, MPI_COMM_WORLD,&request);
      MPI_Irecv(&X2, 1, MPI_INT, (rank-1+size)%size, 101, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, &status);

      // sum up result
      sum += X2;
      X = X2;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0)
  	t2 = seconds();

  printf("Single, non block: Process: %d and the sum is: %d\n",rank,sum);

  if(rank==0)
      printf("NON Blocking Single element: elapsed-time: %f\n\n", t2-t1);
}

// vector being passed, non blocking
void nonblocking_vec(const int rank, const int size, const int N) {

    double t1, t2;

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    	t1 = seconds();

    MPI_Request request;
    MPI_Status status;

    int* Xs = malloc(N*sizeof(int));
    int* X2s = malloc(N*sizeof(int));
    int* sums = malloc(N*sizeof(int));

    // fill vectors
    fill_vec(Xs, rank, N);
    fill_vec(sums, rank, N);
    // each process sends to the next one and receives from the previous one
    for(size_t i=0; i<size-1; i++)
    {

      MPI_Isend(Xs, 1, MPI_INT, (rank+1)%size,101, MPI_COMM_WORLD, &request);

      MPI_Irecv(X2s, 1, MPI_INT, (rank-1+size)%size, 101, MPI_COMM_WORLD, &request);

      MPI_Wait(&request, &status);
      // sum up vectors
      vec_sum(sums,X2s,N);

      // Swapping is only necessary to properly free memory
      swapPP(&Xs, &X2s);

      /*MPI_Isend(&Xs, N, MPI_INT, (rank+1)%size, 101, MPI_COMM_WORLD, &request);
      vec_sum(sums, X2s, N);
	    MPI_Wait(&request, MPI_STATUS_IGNORE);
	    MPI_Recv(&X2s, N, MPI_INT, (rank-1)%size, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);*/
    }

    // free memory
    free(Xs);
    free(X2s);
    free(sums);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    	t2 = seconds();

    printf("VEC: Process: %d and the sum is: %d\n",rank,sums[0]);

    if(rank==0)
      printf("NON Blocking Vector: elapsed-time: %f\n\n", t2-t1);
}


int main(int argc, char* argv[])
{
    size_t N = atoi(argv[1]);

    if(argc < 2)
    	printf("One value must be passed: N");

    int rank, size;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    blocking_single(rank, size);

    MPI_Barrier(MPI_COMM_WORLD);

    nonblocking_single(rank, size);

    MPI_Barrier(MPI_COMM_WORLD);

    blocking_vec(rank, size, N);

    MPI_Barrier(MPI_COMM_WORLD);

    nonblocking_vec(rank, size, N);

    MPI_Finalize();
    return 0;
}
