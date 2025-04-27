export CORES_PER_PROCESS=76
export MPI_CORE_OFFSET=16
# export LD_PRELOAD=/pacific_fs/zjx/demos/mpiperf/libmpiprof.so
mpirun -np 8 --allow-run-as-root --rankfile 1n8p \
    -x UCX_TLS=rc \
	-x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y -x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
	-x LD_PRELOAD \
	-x LD_LIBRARY_PATH \
	-x PATH \
	-x CORES_PER_PROCESS \
	-x MPI_CORE_OFFSET \
	-x MASTER_ADDR=29.204.25.72 \
    -x MASTER_PORT=12355 \
	/pacific_fs/zjx/demos/mpitest/test_base.sh	
#-x THP_MEM_ALLOC_ENABLE=1 \
#-x KML_BLAS_NOT_USE_HBM=0 \
