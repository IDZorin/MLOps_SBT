#!/usr/bin/env bash

# Создадим директорию для plan-файлов
mkdir -p ./assets/trt_plans

# Путь к onnx-файлу
ONNX_MODEL="distilbert.onnx"

SEQ=128

trtexec --onnx=$ONNX_MODEL --minShapes=INPUT_IDS:1x$SEQ,ATTENTION_MASK:1x$SEQ --optShapes=INPUT_IDS:4x$SEQ,ATTENTION_MASK:4x$SEQ --maxShapes=INPUT_IDS:8x$SEQ,ATTENTION_MASK:8x$SEQ --saveEngine=./assets/trt_plans/model_fp32.plan

trtexec --onnx=$ONNX_MODEL --fp16 --minShapes=INPUT_IDS:1x$SEQ,ATTENTION_MASK:1x$SEQ --optShapes=INPUT_IDS:4x$SEQ,ATTENTION_MASK:4x$SEQ --maxShapes=INPUT_IDS:8x$SEQ,ATTENTION_MASK:8x$SEQ --saveEngine=./assets/trt_plans/model_fp16.plan

trtexec --onnx=$ONNX_MODEL --int8 --minShapes=INPUT_IDS:1x$SEQ,ATTENTION_MASK:1x$SEQ --optShapes=INPUT_IDS:4x$SEQ,ATTENTION_MASK:4x$SEQ --maxShapes=INPUT_IDS:8x$SEQ,ATTENTION_MASK:8x$SEQ --saveEngine=./assets/trt_plans/model_int8.plan

trtexec --onnx=$ONNX_MODEL --best --minShapes=INPUT_IDS:1x$SEQ,ATTENTION_MASK:1x$SEQ --optShapes=INPUT_IDS:4x$SEQ,ATTENTION_MASK:4x$SEQ --maxShapes=INPUT_IDS:8x$SEQ,ATTENTION_MASK:8x$SEQ --saveEngine=./assets/trt_plans/model_best.plan
