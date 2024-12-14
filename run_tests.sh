#!/bin/bash

MODELS=("model_ONNX" "model_TRT_FP32" "model_TRT_FP16" "model_TRT_INT8" "model_TRT_BEST")

for MODEL in "${MODELS[@]}"; do
    echo "Running perf_analyzer for $MODEL..."
    perf_analyzer -m $MODEL -u localhost:8701 -i grpc --concurrency-range 1:2 > logs/perf_${MODEL}.log 2>&1 &
done

wait
echo "All perf_analyzer tasks completed."