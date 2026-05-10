from functorch import dim
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
        


class LayerNorm(nn.Module):
    def __init__(self, ebd_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(ebd_dim))
        self.shift = nn.Parameter(torch.zeros(ebd_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        variance = x.var(dim = -1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ebd_dim = cfg['ebd_dim']
        self.layers = nn.Sequential(
            nn.Linear(ebd_dim, ebd_dim * 4),
            nn.GELU(),
            nn.Linear(ebd_dim * 4, ebd_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in = cfg['ebd_dim'],
            d_out = cfg['ebd_dim'],
            dropout = cfg['drop_rate'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            qkv_bias = False
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg['ebd_dim'])
        self.norm2 = nn.LayerNorm(cfg['ebd_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        short_cut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + short_cut

        short_cut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + short_cut
        return x
