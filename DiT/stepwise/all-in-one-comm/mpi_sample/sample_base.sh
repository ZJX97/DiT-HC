#!/usr/bin/env bash

ROOT_DIR="/pacific_ext/wxy/DiT_code"

source ${ROOT_DIR}/env.sh
#export KML_BLAS_NOT_USE_HBM=1
export PYTHONPATH="${PYTHONPATH}:/pacific_ext/wxy/DiT_clean"
export OMP_NUM_THREADS=150
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMPI_MCA_opal_common_verbose=1
export KML_BLAS_THREAD_TYPE=OMP
#export OMPI_MCA_btl_base_verbose=30
#export OMPI_MCA_pml_verbose=10
#export COMM_DEBUG="DEBUG"

echo $PYTHONPATH

python -u "${ROOT_DIR}/sample_ddp.py" \
    --num-fid-samples 50000 \
    --cfg-scale 1.0 \
    --vae mse \
    --pretrain-path ${ROOT_DIR} \
    --model DiT-XL/2 \
    --per-proc-batch-size 1 \
    --num-sampling-steps 10
