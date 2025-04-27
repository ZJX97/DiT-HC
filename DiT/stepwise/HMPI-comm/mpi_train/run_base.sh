#!/usr/bin/env bash

#ROOT_DIR="/pacific_fs/zjx/codes/DiT_clean_0326"

mode=$1 # ["1n4p"|"1n8p"]
batchsize=$2
oneDNN=$3 #[True|False]
gemmProfiler=$4
mpiProfiler=$5
torchProfiler=$6
dataset=$7 #["mini"|"mid"|"all"]

set --

# cpupower frequency-set -f 1.55GHz
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

START_CORE=$((rank* CORES_PER_PROCESS % 608))
OMP_PLACES_LIST=""
for i in $(seq 0 $((CORES_PER_PROCESS-1))); do
    CORE_ID=$((START_CORE + i))
    if [ $((CORE_ID % 38)) -eq 0 ] || [ $((CORE_ID % 38)) -eq 1 ]; then
        continue
    fi
    if [ -z "$OMP_PLACES_LIST" ]; then
        OMP_PLACES_LIST="{${CORE_ID}}"
    else
        OMP_PLACES_LIST="${OMP_PLACES_LIST},{${CORE_ID}}"
    fi
done
OMP_PLACES="$OMP_PLACES_LIST"
export OMP_PLACES

export KML_BLAS_THREAD_TYPE=OMP
export OMP_PROC_BIND=spread
export OMP_NESTED=1
export BLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=active

date=$(date +"%m-%d")
logtime=$(date +"%H-%M")

LIBKBLAS_PATH=/pacific_fs/xuyunpu/data/DiT/gemm/libkblas/omp_0401
export LD_LIBRARY_PATH=$LIBKBLAS_PATH:$LD_LIBRARY_PATH

if [ ${gemmProfiler} = True ]; then
	echo "gemm profiler"
	export LD_PRELOAD=/pacific_fs/xuyunpu/data/DiT/gemm/profiler/libkblasprofiler.so:$LD_PRELOAD
    mkdir -p ${ROOT_DIR}/mpi_train/log_gemm/${date}/${mode}
	export KBLAS_PROFILER_OUTPUT_PATH="${ROOT_DIR}/mpi_train/log_gemm/${date}/${mode}"
fi

if [ ${mpiProfiler} = True ]; then
	echo "mpi profiler"
	export LD_PRELOAD=/pacific_fs/zjx/demos/mpiperf/libmpiprof.so:$LD_PRELOAD
    mkdir -p ${ROOT_DIR}/mpi_train/mpi_log/${date}/${mode}/
	export MPI_PROF_LOGPATH=${ROOT_DIR}/mpi_train/mpi_log/${date}/${mode}/
fi

opt=""
if [ ${torchProfiler} = True ]; then
    mkdir -p ${ROOT_DIR}/mpi_train/profiler_log/${mode}/${date}/${logtime}
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
elif [ ${dataset} = "debug" ]; then
	opt+=" --data-path /pacific_ext/wxy/DiT_dataset/mini-debug"
fi


taskset -c $be-$en python -u "${ROOT_DIR}/mpi_train/train.py" \
    --global-batch-size ${batchsize} \
    --epochs 500 \
    --tp-size 1 \
    --log-every 1 \
    --pretrain-path ${ROOT_DIR} \
    --model DiT-XL/2 \
    --ckpt-every 100 \
    --results-dir checkpoints \
    --profiler-dir ${ROOT_DIR}/mpi_train/profiler_log/${mode}/${date}/${logtime} \
    --global-seed 42 \
    --pre-latent \
    ${opt}
    #--debug
