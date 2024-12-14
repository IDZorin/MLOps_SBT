# torch2onnx.py

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import onnxruntime
from fvcore.nn import FlopCountAnalysis


class MyDistilBERT(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.model.config.hidden_size
        self.fc = nn.Linear(hidden_size, n)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [B, S, H]
        out = self.fc(last_hidden_state)  # [B, S, N]
        return out


def analyze_flops(model, input_ids, attention_mask):
    # Подсчёт FLOP с помощью fvcore
    flops = FlopCountAnalysis(model, (input_ids, attention_mask))
    total_flops = flops.total()
    per_operator_flops = flops.by_operator()

    print("=== FLOP Analysis ===")
    print(f"Total FLOPs: {total_flops}")

    # Операции типа матричного умножения ('mm', 'matmul') - арифметически ограниченные
    # Остальные - считаем условно ограниченными памятью.
    arithmetic_ops = []
    memory_ops = []

    for op, value in per_operator_flops.items():
        # Если в названии оператора присутствует 'mm' или 'matmul', считаем его арифметически ограниченным
        if "mm" in op or "matmul" in op:
            arithmetic_ops.append((op, value))
        else:
            memory_ops.append((op, value))

    print("\nArithmetic-limited operators (example heuristic):")
    for op, val in arithmetic_ops:
        print(f"{op}: {val}")

    print("\nMemory-limited operators (example heuristic):")
    for op, val in memory_ops:
        print(f"{op}: {val}")

    # При увеличении batch size сверх 32 вычислительные затраты на матричные умножения растут значительно
    # и становятся доминирующими.
    threshold_batch_size = 32
    print(f"\nThreshold batch size for arithmetic limitation: {threshold_batch_size}")


def main():
    # Инициализация модели и токенайзера
    model = MyDistilBERT(n=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    input_ids = input_ids.int()
    attention_mask = attention_mask.int()

    # Проверяем вывод модели в PyTorch
    with torch.no_grad():
        torch_output = model(input_ids, attention_mask)

    # Экспорт в ONNX
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        "distilbert.onnx",
        input_names=["INPUT_IDS", "ATTENTION_MASK"],
        output_names=["EMBEDDINGS"],
        dynamic_axes={
            "INPUT_IDS": {0: "BATCH", 1: "SEQUENCE"},
            "ATTENTION_MASK": {0: "BATCH", 1: "SEQUENCE"},
            "EMBEDDINGS": {0: "BATCH", 1: "SEQUENCE"},
        },
        opset_version=19,
    )

    # Санити-чек с помощью onnxruntime
    ort_session = onnxruntime.InferenceSession(
        "distilbert.onnx", providers=["CPUExecutionProvider"]
    )
    ort_inputs = {
        "INPUT_IDS": input_ids.numpy(),
        "ATTENTION_MASK": attention_mask.numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)
    ort_output = torch.tensor(ort_outs[0])

    diff = (torch_output - ort_output).abs().max()
    print("Max diff between torch and onnx:", diff.item())

    # Сохранение токенайзера
    tokenizer.save_pretrained("./assets/tokenizer")

    # Анализ FLOPs
    analyze_flops(model, input_ids, attention_mask)


if __name__ == "__main__":
    main()
