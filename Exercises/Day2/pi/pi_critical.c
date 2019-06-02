#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "cptimer.h"

int main( int argc, char * argv[] ) {

	double start = seconds();
	const unsigned long int N = 1000000000;
	double h = 1./N;
	double pi = 0;
	double h2 = h/2;
	double result = 0;
	double mid;
	
	#pragma omp parallel private(mid, result)
	{
		#pragma omp for schedule(static)
		for(int i = 0; i < N; ++i) {
			mid = (2*i+1)*h2;
			result += 1/(1+mid*mid);
		}
		
		#pragma omp critical
		pi += result;
	}
	
	pi = 4*pi*h;
	printf("%f\n", pi);
	printf("%f \n", seconds()-start);
	
	return 0;
	}
