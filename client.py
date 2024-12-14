import numpy as np
import pandas as pd
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


def call_triton(text: str):
    client = InferenceServerClient(url="0.0.0.0:8700")
    texts = np.array([text], dtype=object)

    input_text = InferInput("TEXTS", texts.shape, np_to_triton_dtype(texts.dtype))
    input_text.set_data_from_numpy(texts)

    outputs = [
        InferRequestedOutput("EMBEDDINGS_ONNX"),
        InferRequestedOutput("EMBEDDINGS_TRT_FP32"),
        InferRequestedOutput("EMBEDDINGS_TRT_FP16"),
        InferRequestedOutput("EMBEDDINGS_TRT_INT8"),
        InferRequestedOutput("EMBEDDINGS_TRT_BEST"),
    ]

    response = client.infer("ensemble", [input_text], outputs=outputs)

    emb_onnx = response.as_numpy("EMBEDDINGS_ONNX")
    emb_fp32 = response.as_numpy("EMBEDDINGS_TRT_FP32")
    emb_fp16 = response.as_numpy("EMBEDDINGS_TRT_FP16")
    emb_int8 = response.as_numpy("EMBEDDINGS_TRT_INT8")
    emb_best = response.as_numpy("EMBEDDINGS_TRT_BEST")

    return emb_onnx, emb_fp32, emb_fp16, emb_int8, emb_best


def check_quality(text: str):
    emb_onnx, emb_fp32, emb_fp16, emb_int8, emb_best = call_triton(text)

    # Считаем среднеквадратичное отклонение
    def diff(a, b):
        return np.mean((a - b) ** 2)

    d_fp32 = diff(emb_onnx, emb_fp32)
    d_fp16 = diff(emb_onnx, emb_fp16)
    d_int8 = diff(emb_onnx, emb_int8)
    d_best = diff(emb_onnx, emb_best)

    return d_fp32, d_fp16, d_int8, d_best


def main():
    df = pd.read_csv("data/data.csv")
    texts = df["text"].tolist()

    diffs_fp32 = []
    diffs_fp16 = []
    diffs_int8 = []
    diffs_best = []

    for t in texts:
        d_fp32, d_fp16, d_int8, d_best = check_quality(t)
        diffs_fp32.append(d_fp32)
        diffs_fp16.append(d_fp16)
        diffs_int8.append(d_int8)
        diffs_best.append(d_best)

    print("Среднее отклонение от ONNX:")
    print("FP32:", np.mean(diffs_fp32))
    print("FP16:", np.mean(diffs_fp16))
    print("INT8:", np.mean(diffs_int8))
    print("BEST:", np.mean(diffs_best))


if __name__ == "__main__":
    main()
