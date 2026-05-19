import torch

class LLMDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_text, tokenizer, tokens_size_per_batch, tride = 1):
        super().__init__()
        self.inputs = []
        self.targets = []

        encoded_tokens_ids = tokenizer.encode(raw_text)
        total_input_len = len(encoded_tokens_ids)
        for i in range(0, total_input_len - tokens_size_per_batch, tride):
            input_chunk = encoded_tokens_ids[i:i + tokens_size_per_batch]
            target_chunk = encoded_tokens_ids[i + 1 : i + tokens_size_per_batch+1]
            self.inputs.append(input_chunk)
            self.targets.append(target_chunk)
        print(f'LLMDataSet init done, total tokens: {total_input_len}, inputs size: {len(self.inputs)}, targets size: {len(self.targets)}')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]