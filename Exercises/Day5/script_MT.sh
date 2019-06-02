cd /home/mcapasso/Parallel_programming/DSSC/Lab/Day5
module load cudatoolkit

for i in 2 4 8 16 32 64 128 256 512 1024 
do
	./matrix_transpose 8192 ${i}>> output_MT.dat
done

