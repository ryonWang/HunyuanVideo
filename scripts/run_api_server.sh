#!/bin/bash

# Redis配置
export REDIS_HOST="localhost"
export REDIS_PORT="6379"

# 其他配置
export CUDA_VISIBLE_DEVICES=0
export SERVER_NAME="0.0.0.0"
export SERVER_PORT="5000"
export MODEL_BASE="/path/to/your/model/ckpts"

# 启动API服务器
python api/main.py 