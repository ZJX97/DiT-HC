#!/bin/bash

ROOT_DIR=$(dirname $(pwd))
echo $ROOT_DIR
source ${ROOT_DIR}/env.sh
mpirun 	--allow-run-as-root -np 4 \
	--rankfile ${ROOT_DIR}/mpi_train/rankfiles/1n4p \
	--mca coll ^ucg \
	-x UCX_TLS=rc \
	-x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y \
	-x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
	${ROOT_DIR}/mpi_train/run_base.sh
	#--hostfile ${ROOT_DIR}/mpi_train/rankfiles/hostfile2n \
