CC=gcc

all: pi_reduction pi_critical pi_atomic

pi_reduction: 
	${CC} -std=c99 -fopenmp pi_reduction.c -o pi_reduction
	
pi_critical: 
	${CC} -std=c99 -fopenmp  pi_critical.c -o pi_critical
		
pi_atomic: 
	${CC} -std=c99 -fopenmp pi_atomic.c -o pi_atomic
			
clean: 
	rm -rf pi_reduction pi_atomic pi_critical
