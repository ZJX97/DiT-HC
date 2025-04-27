#!/usr/bin/env bash

source /pacific_fs/zjx/workspace/kpops_linear/scripts/env.sh
taskset -c 0-151 /root/tmp/anaconda3/envs/DiT/python kpLayernorm.py


