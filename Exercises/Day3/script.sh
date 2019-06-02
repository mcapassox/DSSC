module load openmpi
cd /home/mcapasso/Parallel_programming/DSSC/Lab/Day3/

for i in 1 2 4 8 16 20 30 40
do
	mpirun -np ${i} ./midpoint_pi >> output.dat
done
