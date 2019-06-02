CC=mpicc

all: ring dist_mat

ring: 
	${CC} -std=c99 ring.c utils.c cptimer.c -o ring
	
dist_mat: 
	${CC} -std=c99  dist_mat.c utils.c cptimer.c -o dist_mat
		
clean:
	rm -rf dist_mat ring
