#include <stdlib.h>
#include <stdio.h>

int main( int argc, char * argv[] ) {

	const long unsigned int N = 1000000000;
	double h = 1./N;
	double pi = 0;
	double h2 = h/2;
	double mid;
	double result;

	for(int i = 0; i < N; i++)
	{
		mid = (2*i+1)*h2;
		result = 1/(1+mid*mid);
		pi += result;

	}
	pi = 4*pi*h;
	printf("%f", pi);
	return 0;
}
