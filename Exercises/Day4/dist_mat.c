#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "utils.h"
#include "cptimer.h"

#define REPETITIONS 5

// blocking version
void blocking_dist(int** dist_mat_row, const int* chunk_size, const size_t n_cols, const size_t rank, const size_t size) {
    
    // number of rows of each slice
    int n_rows = chunk_size[rank];
	
    // print on screen if n < 10
    if(n_cols < 10) {
    	// 0th process receives, sets and prints
        if (rank == 0) {
            // 1's offset wrt to the left of the matrix
            size_t offset = 0;
            // set diagonal element(s) of the 0's slice
            set_one(dist_mat_row, n_rows, offset);
	    // print matrix
            print_matrix(dist_mat_row, n_rows, n_cols);
            // chunk_size[0] is the offset of the 1st slice (counting from 0)
            offset = chunk_size[0];
            // receive from all other processes
            for (size_t i = 1; i < size; i++) {
                // receive slices a row at time
                for (size_t j = 0; j < chunk_size[i]; j++) {
                    // receive from ith process
                    MPI_Recv(dist_mat_row[j], n_cols, MPI_INT, i, 101, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    // set 1 in the right position per each row
                    set_one_vect(dist_mat_row[j], offset+j);
                    // print
                    print_vector(dist_mat_row[j], n_cols);
                }
                // increase offset
                offset += chunk_size[i];
            }

        }
        else {
            for (size_t i = 0; i < n_rows; i++) {
            // all other processes send
                MPI_Send(dist_mat_row[i], n_cols, MPI_INT, 0, 101, MPI_COMM_WORLD);
            }
        }
    }
    else {
    	// 0th process receives, sents and prints on file
        if (rank == 0) {
            // prepare file
            FILE* data_file;
            data_file=fopen("data.dat","wb");
            
            // 0th process has 0 offset
            size_t offset = 0;
            set_one(dist_mat_row, n_rows, offset);
            // print on file
            for (size_t i = 0; i < n_rows; i++) {
                fwrite(dist_mat_row[i], sizeof(int), n_cols, data_file);
            }
            offset = chunk_size[0];
            // receive from all other processes
            for (size_t i = 1; i < size; i++) {
                for (size_t j = 0; j < chunk_size[i]; j++) {
                    MPI_Recv(dist_mat_row[j], n_cols, MPI_INT, i, 101, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    // set one in right position and print on file
                    set_one_vect(dist_mat_row[j], offset+j);
                    fwrite(dist_mat_row[j], sizeof(int), n_cols, data_file);
                }
                offset += chunk_size[i];
            }

            fclose(data_file);
 	    // read the matrix from file and print it (just for checking)
 	    //int* complete_mat = (int*) malloc(sizeof(int)*n_cols*n_cols);
            //data_file = fopen("data.dat","r");
            //fread(complete_mat,sizeof(int),n_cols*n_cols,data_file);
            //print_vectMat(complete_mat,n_cols);
            //free(complete_mat);
            //fclose(data_file);
        }
        else {
            // all processes send to 0
            for (size_t i = 0; i < n_rows; i++) {
                MPI_Send(dist_mat_row[i], n_cols, MPI_INT, 0, 101, MPI_COMM_WORLD);
            }
        }
    }
}

void non_blocking_dist(int** dist_mat_row, int* chunk_size, const size_t n_cols, const size_t rank, size_t size) {

    int n_rows = chunk_size[rank];
    MPI_Request request;
    MPI_Status status;
    
    // chunk_size[0] is the max number of rows of all the slices
    int max = chunk_size[0];
    int **support = callocate_matrix(max, n_cols);

    if(n_cols < 10) {
        if (rank == 0) {
            size_t offset = 0;
            // Receive from each process
            for (size_t i = 1; i < size; i++) {
                // Receive each slice
                for (size_t j = 0; j < chunk_size[i]; j++) {
                    MPI_Irecv(support[j], n_cols, MPI_INT, i, 101, MPI_COMM_WORLD, &request);
                }
                // while receiving the ith slice, write and print the (i-1)th slice
                for (size_t k = 0; k < chunk_size[i-1]; k++) {
                    set_one_vect(dist_mat_row[k], offset+k);
                    print_vector(dist_mat_row[k], n_cols);
                }

                MPI_Wait(&request,&status);
                // copy the received slice in dist_mat_row
                for (size_t j = 0; j < chunk_size[i]; j++) {
                    copy(support[j],dist_mat_row[j], n_cols);
                }

                offset += chunk_size[i-1];
            }
            
            // set and print last slice
            for (size_t k = 0; k < chunk_size[size-1]; k++) {
                set_one_vect(dist_mat_row[k], offset+k);
                print_vector(dist_mat_row[k], n_cols);
            }
        }
        else {
            // all processes send to 0
            for (size_t i = 0; i < n_rows; i++) {
                MPI_Isend(dist_mat_row[i], n_cols, MPI_INT, 0, 101, MPI_COMM_WORLD, &request);
            }
        }
    }
    else {
        if (rank == 0) {
            size_t offset = 0;

            FILE* data_file;
            data_file=fopen("data.dat","wb");

            // Receive from each process
            for (size_t i = 1; i < size; i++) {
                // Receive each slice
                for (size_t j = 0; j < chunk_size[i]; j++) {
                    MPI_Irecv(support[j], n_cols, MPI_INT, i, 101, MPI_COMM_WORLD, &request);
                }
                // while receiving, set and print
                for (size_t k = 0; k < chunk_size[i-1]; k++) {
                    set_one_vect(dist_mat_row[k], offset+k);
                    fwrite(dist_mat_row[k], sizeof(int), n_cols, data_file);
                }

                MPI_Wait(&request,&status);
                for (size_t j = 0; j < chunk_size[i]; j++) {
                    copy(support[j],dist_mat_row[j], n_cols);
                }

                offset += chunk_size[i-1];
            }
            
            // set and print last slice
            for (size_t k = 0; k < chunk_size[size-1]; k++) {
                set_one_vect(dist_mat_row[k], offset+k);
                fwrite(dist_mat_row[k], sizeof(int), n_cols, data_file);
            }
            
            fclose(data_file);

	    //just for checking
            /*int* complete_mat = (int*) malloc(sizeof(int)*n_cols*n_cols);
            data_file = fopen("data.dat","r");
            fread(complete_mat,sizeof(int),n_cols*n_cols,data_file);
            print_vectMat(complete_mat,n_cols);
            free(complete_mat);
            fclose(data_file);*/
        }
        else {
        // all processes send to 0
            for (size_t i = 0; i < n_rows; i++) {
                MPI_Isend(dist_mat_row[i], n_cols, MPI_INT, 0, 101, MPI_COMM_WORLD, &request);
            }
        }
    }
    deallocate_matrix(support, max);
  }

    int main (int argc, char* argv[])
    {
        double begin, end;

        unsigned int rank, size, X, sum, i, N;

        if (argc < 2) {
          printf("Matrix dimension N must be inserted. RETURNING\n");
          return 1;
        }

        MPI_Init (&argc, &argv);

        //Get process ID
        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
        //Get processes Number
        MPI_Comm_size (MPI_COMM_WORLD, &size);

        MPI_Barrier(MPI_COMM_WORLD);


        N = atoi(argv[1]);
	if(rank == 0)
        printf("%d,", size);

        // Row distribution
	
	// split the matrix in size different (row) slices
        int min_chunk = N/size, chunk_rem = N%size;
        int chunk_size[size];

        for ( i = 0; i < size; i++) {
            chunk_size[i] = min_chunk;
        }

        for ( i = 0; i < chunk_rem; i++) {
            chunk_size[i]++;
        }


        int n_cols = N;
        int n_rows = chunk_size[rank];
        
        // declare (row) slices
        int** dist_mat_row;
        double accum = 0;

        // ------- blocking -------
        for (size_t i = 0; i < REPETITIONS; i++) {
          MPI_Barrier(MPI_COMM_WORLD);
          begin = seconds();
          // each thread allocates its own slice (and fills it with zeros)
          dist_mat_row = callocate_matrix(n_rows, n_cols);
          // function call
          blocking_dist(dist_mat_row, chunk_size, n_cols, rank, size);
          MPI_Barrier(MPI_COMM_WORLD);
          end = seconds();
          accum += end-begin;
        }
	
	// print time of blocking version
        if (rank == 0) {
            printf("%f,", (end-begin)/REPETITIONS);
        }
        accum = 0;
        
        // ------- non blocking -------
        for (size_t i = 0; i < REPETITIONS; i++) {
          MPI_Barrier(MPI_COMM_WORLD);
          begin = seconds();
          // each thread allocates its own slice (and fills it with zeros)
          dist_mat_row = callocate_matrix(n_rows, n_cols);
	  // function call
          non_blocking_dist(dist_mat_row, chunk_size, n_cols, rank, size);
          MPI_Barrier(MPI_COMM_WORLD);
          end = seconds();
          accum += end-begin;
        }
	
	// print time of non blocking version
        if (rank == 0) {
            printf("%f\n", (end-begin)/REPETITIONS);
        }
	
	// free memory
        deallocate_matrix(dist_mat_row, n_rows);

        MPI_Finalize();

        return 0;
    }
