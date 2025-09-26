import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import torch.nn.functional as F
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


warnings.filterwarnings("ignore")


# Embeddings
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)#输入初始化之后会生成一个(vocab,d_model)大小的矩阵，为每一个索引生成了一个对应的向量。
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# #调试
# vocab = 10
# d_model = 16
# embedding = Embeddings(d_model, vocab)
# # 输入示例：batch_size=2, seq_len=5
# x = torch.tensor([[1, 2, 3, 4, 5],
#                   [0, 2, 3, 1, 9]])
# out = embedding(x)
# print("输出形状:", out.shape)      # [2, 5, 16]
# print("输出示例:", out[0,0])      # 第一个序列第一个词的 embedding

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)#这里是位置编码的初始化
        #表示最多生成max_len个token生成维度为d_model的位置编码。
        position = torch.arange(0, max_len).unsqueeze(1)#(max_len,)->(max_len,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)#
        pe = pe.unsqueeze(0)#(max_len, d_model)->(1,max_len, d_model)
        #
        self.register_buffer("pe", pe)#不随模型一块训练更新参数

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
# #调试
# d_model = 16
# dropout = 0.1
# batch_size = 2
# seq_len = 10

# pos_encoder = PositionalEncoding(d_model, dropout)
# x = torch.zeros(batch_size, seq_len, d_model)  # 模拟 embedding
# out = pos_encoder(x)

# print("输出形状:", out.shape)  # [2,10,16]
# print("前两行位置编码示例:\n", pos_encoder.pe[0, :2])

# Multi-Head Attention
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None #注意力分数，后边需要通过attention函数计算
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)#batch_size

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

# #调试
# batch = 2
# seq_len = 5
# d_model = 16
# h = 4

# mha = MultiHeadedAttention(h, d_model)
# x = torch.randn(batch, seq_len, d_model)
# out = mha(x, x, x)
# print(out.shape)  # 应该是 [2, 5, 16]
# print(mha.attn.shape)  # 应该是 [2, 4, 5, 5] 注意力分数 长度为(batch_size, num_heads, seq_len, seq_len)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)#若加上unbiased=False，则和torch.nn.LayerNorm结果一致
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
#调试
# 参数
# features = 4
# layernorm_custom = LayerNorm(features)
# layernorm_torch = nn.LayerNorm(features)

# # 模拟输入
# x = torch.randn(2, 3, features)  # batch=2, seq_len=3, dim=4

# y_custom = layernorm_custom(x)
# y_torch = layernorm_torch(x)

# print("输入:\n", x)
# print("\n自定义 LayerNorm:\n", y_custom)
# print("\nPyTorch LayerNorm:\n", y_torch)

# 差异
#这里之所以有差异，是因为PyTorch 里：
# torch.std(input, unbiased=True)（默认是 True)
# unbiased=True → 样本标准差，分母是 N-1。
# unbiased=False → 总体标准差，分母是 N。
#  而我们自定义的 LayerNorm 里，std = x.std(-1, keepdim=True)默认为unbiased=True。
# 而torch.nn.LayerNorm为unbiased=False。


# SublayerConnection
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

# PositionwiseFeedForward
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu())) 

# EncoderLayer
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
# #调试
# h=8
# d_model=16
# d_ff=64
# dropout=0.1
# N=6
# c = copy.deepcopy
# attn = MultiHeadedAttention(h, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# position = PositionalEncoding(d_model, dropout)
# model=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
# x = torch.randn(2, 10, d_model)  # batch=2, seq_len=10
# mask = None
# out = model(x, mask)
# print(model)
# print("输入 shape:", x.shape)     # [2, 10, 16]
# print("输出 shape:", out.shape)   # [2, 10, 16]


# DecoderLayer

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# Decoder
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# 生成上三角为False的矩阵, 即subsequent mask
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).bool() #会生成上三角为True的矩阵

    return subsequent_mask ==False  #取反，使得上三角为False


# 简单的测试
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src)
        )
        prob = test_model.generator(out[:, -1])
        # print(prob)
        _, next_word = torch.max(prob, dim=1)#返回最大值和最大值所在的索引，我们只需要索引
        print(next_word)
        next_word = next_word[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()

# run_tests()

# temp=subsequent_mask(5)
# print(temp)

