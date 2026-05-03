from nltk.tokenize import word_tokenize
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

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


class CausalAttention(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob, contxt_length, bias=True):
        super().__init__()
        self.W_key = nn.Linear(in_dim, out_dim, bias=bias)
        self.W_value = nn.Linear(in_dim, out_dim, bias=bias)
        self.W_query = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_dim = out_dim
        self.mask = torch.triu(torch.ones(contxt_length, contxt_length), diagonal=1)
        # self.register_buffer()

    def forward(self, x):
        batch_size, token_size, d_in = x.shape

        query = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = query @ keys.transpose(1, 2)
        attention_scores.maskd_fill_(self.mask.bool(), -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        x_context = attention_weights @ values
        return x_context