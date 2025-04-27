#!/usr/bin/env bash

#ROOT_DIR="/pacific_fs/zjx/codes/DiT_clean_0326"

mode="1n4p" # ["1n4p"|"1n8p"]
batchsize=28
oneDNN=True #[True|False]
gemmProfiler=False
mpiProfiler=False
torchProfiler=True
dataset="mini" #["mini"|"mid"|"all"]

cpupower frequency-set -f 1.55GHz

ROOT_DIR=$(dirname "$(pwd)")
echo $ROOT_DIR

sh /pacific_fs/xuyunpu/shell/reserve_hugepage.sh 2044 2044
source ${ROOT_DIR}/env.sh

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}"


rank=$OMPI_COMM_WORLD_RANK
if [ ${mode} = "1n4p" ]; then
	be=$((rank%4*152))
	en=$((be+151))
	export OMP_NUM_THREADS=144
	export THREADNUM_L1=4
elif [ ${mode} = "1n8p" ]; then
	be=$((rank%8*76))
	en=$((be+75))
	export OMP_NUM_THREADS=72
	export THREADNUM_L1=2
fi
export THREADNUM_L2=36

export OMP_PLACES=cores
export KML_BLAS_THREAD_TYPE=OMP
export OMP_PROC_BIND=spread
export OMP_NESTED=1
export BLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=active


LIBKBLAS_PATH=/pacific_fs/xuyunpu/data/DiT/gemm/libkblas/omp_0326
export LD_LIBRARY_PATH=$LIBKBLAS_PATH:$LD_LIBRARY_PATH

if [ ${gemmProfiler} = True ]; then
	echo "gemm profiler"
	export LD_PRELOAD=/pacific_fs/xuyunpu/data/DiT/gemm/profiler/libkblasprofiler.so
	export KBLAS_PROFILER_OUTPUT_PATH="${ROOT_DIR}/mpi_train/log_gemm"
fi

if [ ${mpiProfiler} = True ]; then
	echo "mpi profiler"
	export LD_PRELOAD=/pacific_fs/zjx/demos/mpiperf/libmpiprof.so
	export MPI_PROF_LOGPATH=./mpi_log/4-2/8n8p/
fi

	

opt=""
if [ ${torchProfiler} = True ]; then
	echo "torchProfiler"
	opt+=" --debug"
fi

if [ ${oneDNN} = False ]; then
	opt+=" --banOneDNN"
	echo "ban onednn"
fi

if [ ${dataset} = "mini" ]; then
	opt+=" --data-path /pacific_ext/wxy/DiT_dataset/imagenet-mini-latents"
elif [ ${dataset} = "mid" ]; then
	opt+=" --data-path /pacific_ext/wxy/DiT_dataset/imagenet-mid-latents"
elif [ ${dataset} = "all" ]; then
	opt+=" --data-path /pacific_ext/wxy/DiT_dataset/imagenet-all-latents"
fi
              
taskset -c $be-$en python -u "${ROOT_DIR}/mpi_train/train.py" \
    --global-batch-size ${batchsize} \
    --epochs 500 \
    --tp-size 1 \
    --log-every 5 \
    --pretrain-path ${ROOT_DIR} \
    --model DiT-XL/2 \
    --ckpt-every 100 \
    --results-dir checkpoints \
    --global-seed 42 \
    --pre-latent \
    ${opt}
    #--debug
