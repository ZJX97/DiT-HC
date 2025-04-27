export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$1
taskset -c 0-$1 python $2

