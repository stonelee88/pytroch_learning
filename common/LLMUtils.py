from nltk.tokenize import word_tokenize
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetv1(Dataset):
    def __init__(self, raw_text, max_len=4, stride = 1):
        self.input_ids = []
        self.target_ids = []
        self.tiktokenTokenier = tiktoken.get_encoding('gpt2')
        raw_text_tokens = self.tiktokenTokenier.encode(raw_text)

        for i in range(0, len(raw_text_tokens) - max_len, stride):
            inputs = raw_text_tokens[i: i + max_len]
            targets = raw_text_tokens[i + 1 : i + max_len + 1]
            self.input_ids.append(torch.Tensor(inputs))
            self.target_ids.append(torch.Tensor(targets))

        print('GPTDatasetv1 init completed')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.target_ids[item]