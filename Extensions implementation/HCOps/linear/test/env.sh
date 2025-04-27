#source /pacific_fs/xuyunpu/shell/hpckit_env_init.sh
source /pacific_fs/xuyunpu/shell/hpckit_env_init.sh
#export LIBKBLAS_PATH=/pacific_fs/xuyunpu/data/DiT/gemm/libkblas/omp_0324_later
export LIBKBLAS_PATH=/pacific_fs/xuyunpu/data/DiT/gemm/libkblas/omp_0326
export LD_LIBRARY_PATH=$LIBKBLAS_PATH:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=72
export KML_BLAS_THREAD_TYPE=OMP
export BLAS_NUM_THREADS=1
export OMP_NESTED=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export OMP_DISPLAY_AFFINITY=1

export THREADNUM_L1=2
export THREADNUM_L2=36
