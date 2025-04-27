#!/bin/bash

# ROOT_DIR="/pacific_fs/xuyunpu/data/DiT/DiT_code"
ROOT_DIR="/pacific_ext/wxy/DiT_clean"
source ${ROOT_DIR}/env.sh
#export KML_BLAS_NOT_USE_HBM=1

#export PYTHONPATH="${ROOTDIR}"

mpirun 	--allow-run-as-root -np 4 \
	--rankfile ${ROOT_DIR}/rankfiles/rankfile_4rank_54 \
	--mca coll ^ucg \
	-x UCX_TLS=rc \
	-x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y \
	-x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
    	${ROOT_DIR}/mpi_sample/sample_base.sh

	# --rankfile '/pacific_fs/xuyunpu/data/DiT/DiT_code/rankfile' \
