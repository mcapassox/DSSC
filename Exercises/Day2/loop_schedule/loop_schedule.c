#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void print_usage( int * a, int N, int nthreads ) {

  int tid, i;
  for( tid = 0; tid < nthreads; ++tid ) {

    fprintf( stdout, "%d: ", tid );

    for( i = 0; i < N; ++i ) {

      if( a[ i ] == tid) fprintf( stdout, "*" );
      else fprintf( stdout, " ");
    }
    printf("\n");
  }
}

int main( int argc, char * argv[] ) {

	const int N = 100;
	int a[N];
  	int thread_id = 0;
	int nthreads;
	int chunk;
	omp_sched_t kind;
	
	#pragma omp parallel
	{
	
	nthreads = omp_get_num_threads();
	omp_get_schedule(&kind,&chunk);
	
	#pragma omp for schedule(runtime)
  	for(int i = 0; i < N; ++i) {
  	  thread_id = omp_get_thread_num();
  	  a[i] = thread_id;
  	}

	#pragma omp single
	print_usage(a,N,nthreads);

	}

return 0;
}
