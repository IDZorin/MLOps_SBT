import numpy as np
import transformers
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/assets/tokenizer", local_files_only=True
        )
        self.max_length = 128

    def tokenize(self, texts):
        encoded = self.tokenizer(
            texts, padding="max_length", max_length=self.max_length, truncation=True
        )
        input_ids = np.array(encoded["input_ids"], dtype=np.int32)
        attention_mask = np.array(encoded["attention_mask"], dtype=np.int32)
        return input_ids, attention_mask

    def execute(self, requests):
        responses = []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [el.decode("utf-8") for el in texts]

            input_ids, attention_mask = self.tokenize(texts)

            out_input_ids = pb_utils.Tensor("INPUT_IDS", input_ids)
            out_attention_mask = pb_utils.Tensor("ATTENTION_MASK", attention_mask)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_input_ids, out_attention_mask]
                )
            )

        return responses
