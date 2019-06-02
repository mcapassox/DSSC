cd /home/mcapasso/Parallel_programming/DSSC/Lab/Day5/

module load cudatoolkit

for i in 4 8 16 32 64 128 256 512 1024
do
	./matrix_matrix 2048 ${i} >> output_MM.dat

done 



