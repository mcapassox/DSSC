module load openmpi
cd /home/mcapasso/Parallel_programming/DSSC/Lab/Day4/

for i in 2 4 8 16 20 30 40
do
	mpirun -np ${i} ./dist_mat 10000 >> output_dist.out
done
