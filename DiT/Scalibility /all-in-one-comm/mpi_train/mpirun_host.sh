#!/bin/bash

ROOT_DIR=$(dirname $(pwd))
echo $ROOT_DIR
export CORES_PER_PROCESS=152
export MPI_CORE_OFFSET=0
source ${ROOT_DIR}/env.sh

hostfile=${ROOT_DIR}/mpi_train/rankfiles/node2xslot4
numnodes=$(wc -l < $hostfile)
batchsize=$((numnodes*28))
# batchsize=112
numrank=$((numnodes*4))
masteraddr=$(hostname -I | awk '{print $1}')
echo $masteraddr
mpirun 	--allow-run-as-root -np $numrank \
	--hostfile $hostfile \
	--mca coll ^ucg \
	-x UCX_TLS=rc \
	-x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y \
	-x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
	-x MASTER_ADDR=$masteraddr \
    -x MASTER_PORT=29502 \
   	-x CORES_PER_PROCESS \
    -x MPI_CORE_OFFSET \
	${ROOT_DIR}/mpi_train/run_base.sh $batchsize
