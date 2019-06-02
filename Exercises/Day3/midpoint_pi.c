#include <stdio.h>
#include <mpi.h>
#include "cptimer.h"


int main (int argc, char* argv[])
{
    double begin;

    const unsigned long int N = 1000000000;
    int rank, size, error, i;
    double pi=0.0, result=0.0, sum=0.0, x2, end = 0.0;
    double d = 1/(double)N;
    double d2 = d/2;

    MPI_Init (&argc, &argv);

    //Get process ID
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    //Get processes Number
    MPI_Comm_size (MPI_COMM_WORLD, &size);
	  if(rank == 0)
	    begin = seconds();

    //Each process caculates a part of the sum
    for (i=N/size*rank; i<N/size*(rank+1); i++)
    {
        x2 = (2*i+1)*d2;
        result+= 1/(1+x2*x2);
    }

    //Sum up all results
    MPI_Reduce(&result, &sum, 1, MPI_DOUBLE, MPI_SUM, size-1, MPI_COMM_WORLD);

    //Send it to 0
    if(rank == size-1)
    	 MPI_Send(&sum, 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);

    //Calculate and print PI
    if (rank==0)
    {
	     MPI_Recv(&sum, 1, MPI_DOUBLE, size-1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	     pi=4*d*sum;
       //printf("PI=%lf, ", size, pi);

    }

    if(rank == 0)
      end=seconds();

    if (rank==0)
	    printf("%d, %f\n",size, end-begin);

    MPI_Finalize();

    return 0;
}
