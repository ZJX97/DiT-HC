#!/usr/bin/env bash

#source /pacific_fs/xuyunpu/data/DiT/gemm/nested/env.sh
source /pacific_fs/zjx/workspace/kpops_linear/scripts/env.sh
LD_PRELOAD=/pacific_fs/xuyunpu/data/DiT/gemm/profiler/libkblasprofiler.so taskset -c 456-607 /pacific_fs/xuyunpu/app/anaconda3/envs/DiT/bin/python /pacific_fs/zjx/workspace/kpops_linear/test/kpnn.py


