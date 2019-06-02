module load openmpi

cd /home/mcapasso/Parallel_programming/DSSC/Lab/Day2/loop_schedule

export OMP_NUM_THREADS=1
./loop_schedule >> output.out

echo "\n" >> output.out

export OMP_NUM_THREADS=10
export OMP_SCHEDULE="STATIC,1"
./loop_schedule >> output.out

echo "\n" >> output.out

export OMP_NUM_THREADS=10
export OMP_SCHEDULE="STATIC, 10"
./loop_schedule >> output.out

echo "\n" >> output.out

export OMP_NUM_THREADS=10
export OMP_SCHEDULE="DYNAMIC,1"
./loop_schedule >> output.out

echo "\n" >> output.out

export OMP_NUM_THREADS=10
export OMP_SCHEDULE="DYNAMIC,10"
./loop_schedule >> output.out
