#!/usr/bin/env bash

cd code

ARGS=$@
GPU_ID=0
CMD="python run.py $ARGS"

CUDA_VISIBLE_DEVICES=$GPU_ID $CMD
