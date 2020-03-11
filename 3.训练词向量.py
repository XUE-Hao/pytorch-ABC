import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity


USE_CUDA = torch.cuda.is_available()

# 为保证可复现性，给定随机种子
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

# 设定一些超参数
C = 2  # 考虑周围3个单词
K = 10  # 负例个数
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 32
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 25


# 预处理
def word_tokensize(text):
    return text.split()


with open('text8.train.txt', 'r') as fin:
    text = fin.read()

text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)  # 论文要求
word_freqs = word_freqs / np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word)


# 实现Dataloader
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        # 这个数据集医用有多少个item
        return len(self.text_encoded)

    def __getitem__(self, idx):
        # 返回torch的tensor
        center_word = self.text_encoded[idx]  # 中心词
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1))  # 窗口内单词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 取余，防止超出文本长度
        pos_words = self.text_encoded[pos_indices]  # 周围单词
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)  # 副例采样单词
        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# print(next(iter(dataloader)))


# 定义PyTorch模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embd_size = embed_size

        initrange = 0.5 / self.embd_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embd_size)
        self.in_embed.weight.data.uniform_(initrange, initrange)
        self.out_embed = nn.Embedding(self.vocab_size, self.embd_size)
        self.out_embed.weight.data.uniform_(initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_label: [batch_size]
        # pos_labels: [batch_size, (window_size * 2)]
        # neg_lables: [batch_size, (window_size * 2 * K)]
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window_size * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_size, (window_size * 2)] bmm: Batch Matrix Multi 批次中每个矩阵分别相乘
        neg_dot = torch.bmm(pos_embedding, -input_embedding).squeeze(2)  # [batch_size, (window_size * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos+log_neg
        return -loss  # [batch_size] 这里已经定义好了loss_function

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        # print(input_labels, pos_labels, neg_labels)
        # if i > 5:
        #     break
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('epoch', e, 'iteration', i, loss.item())
