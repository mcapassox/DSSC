CC=nvcc

all: matrix_transpose matrix_matrix array_reversal

matrix_transpose: 
	${CC} matrix_transpose.cu utils.cu -o matrix_transpose
	
matrix_matrix: 
	${CC} matrix_matrix.cu utils.cu -o matrix_matrix
		
array_reversal: 
	${CC} array_reversal.cu -o array_reversal
clean:
	rm -rf matrix_matrix matrix_transpose array_reversal
