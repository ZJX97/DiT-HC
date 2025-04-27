#!/bin/bash

numnode=1
mode="1n4p" # ["1n4p"|"1n8p"]
oneDNN=True #[True|False]
gemmProfiler=False
mpiProfiler=False
torchProfiler=True
dataset="mini" # ["mini"|"mid"|"all" | "debug"]
rankfile=$mode
masteraddr="localhost"

ROOT_DIR=$(dirname $(pwd))
echo $ROOT_DIR
if [ ${mode} = "1n4p" ]; then
	export CORES_PER_PROCESS=152
	batchsize=$((28*numnode))
	np=$((4*numnode))
else
	export CORES_PER_PROCESS=76
	batchsize=$((32*numnode))
	np=$((8*numnode))
fi

export MPI_CORE_OFFSET=0
source ${ROOT_DIR}/env.sh


echo "numnode:${numnode} np:${np} mode:${mode} batchsize:${batchsize} oneDNN:${oneDNN} gemmProfiler:${gemmProfiler} mpiProfiler:${mpiProfiler} torchProfiler:${torchProfiler} dataset:${dataset}"

mpirun 	--allow-run-as-root -np $np \
	--rankfile ${ROOT_DIR}/mpi_train/rankfiles/${rankfile} \
	--mca coll ^ucg \
	-x UCX_TLS=rc \
	-x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y \
	-x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
	-x MASTER_ADDR=${masteraddr} \
    -x MASTER_PORT=29500 \
    -x CORES_PER_PROCESS \
    -x MPI_CORE_OFFSET \
	${ROOT_DIR}/mpi_train/run_base.sh $mode $batchsize $oneDNN $gemmProfiler $mpiProfiler $torchProfiler $dataset