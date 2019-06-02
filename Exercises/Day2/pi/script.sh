module load openmpi
cd /home/mcapasso/Parallel_programming/DSSC/Lab/Day2/pi

for i in 1 2 4 8 16 20 30 40
do
	
	export OMP_NUM_THREADS=${i}
	./pi_atomic >> output.out
done

for i in 1 2 4 8 16 20 30 40
do
	
	export OMP_NUM_THREADS=${i}
	./pi_reduction >> output.out
done

for i in 1 2 4 8 16 20 30 40
do
	
	export OMP_NUM_THREADS=${i}
	./pi_critical >> output.out
done


