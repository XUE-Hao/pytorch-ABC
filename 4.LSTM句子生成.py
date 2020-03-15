import torchtext  # 建议自己写，不用这个包
from torchtext.vocab import Vectors
import torch
import numpy as np
import random
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 100
MAX_VOCAB_SIZE = 50000
NUM_EPOCHS = 2
GRAD_CLIP = 5.0

TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path='.',
                                                  train='text8.train.txt',
                                                  validation='text8.dev.txt',
                                                  test='text8.test.txt',
                                                  text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
# print(TEXT.vocab.stoi['<unk>'])
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test),
                                                                     batch_size=BATCH_SIZE,
                                                                     device=device,
                                                                     bptt_len=50,
                                                                     repeat=False,
                                                                     shuffle=True)
VOCAB_SIZE = len(TEXT.vocab)
# it = iter(train_iter)
# batch = next(it)
# print(' '.join(TEXT.vocab.itos[i] for i in batch.text[:, 0].data.cpu()))
# print()
# print(' '.join(TEXT.vocab.itos[i] for i in batch.target[:, 0].data.cpu()))
# for i in range(5):
#     batch = next(it)
#     print(i)
#     print(' '.join(TEXT.vocab.itos[i] for i in batch.text[:, 0].data.cpu()))
#     print()
#     print(' '.join(TEXT.vocab.itos[i] for i in batch.target[:, 0].data.cpu()))


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embeding_size, hidden_size):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embeding_size)
        self.lstm = nn.LSTM(embeding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, text, hidden):
        # forward pass
        # text: seq_length * batch_size
        emb = self.embed(text)  # seq_length * batch_size * embed_size
        output, hidden = self.lstm(emb, hidden)
        # output: seq_length * batch_size * hidden_size
        # hidden: (1 * batch_size * hidden_size, 1 * batch_size * hidden_size)
        out_vocab = self.linear(output.view(-1, output.shape[2]))  # view把前两个纬度拼到一起
        # out_vocab: (seq_length * batch_size) * vocab_size
        out_vocab = out_vocab.view(output.size(0), output.size(1), out_vocab.size(-1))
        return out_vocab, hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())  # 若在GPU上，拿到GPU的tensor。若在CPU上，拿到CPU的tensor
        return (weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad),
                weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad))


model = RNNModel(vocab_size=VOCAB_SIZE,
                 embeding_size=EMBEDDING_SIZE,
                 hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    model = model.to(device)
# print(next(model.parameters()))


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()  # 和前面的计算图截断，只保留值
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model, data):
    model.eval()
    total_loss = 0.
    total_count = 0.
    it = iter(data)
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad=False)
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))  # batch_size * target_class_dim, batch_size
            total_loss = loss.item() * np.multiply(*data.size())
            total_count = np.multiply(*data.size())
    loss = total_loss/total_count
    model.train()
    return loss


loss_fn = nn.CrossEntropyLoss()
learn_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
val_losses = []
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)  # 每次调用，降低学习率一半

for epoch in range(NUM_EPOCHS):
    model.train()  # 每次开始训练前都写
    it = iter(train_iter)
    hidden = model.init_hidden(BATCH_SIZE)
    for i, batch in enumerate(it):
        data, target = batch.text, batch.target
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)  # 只有训练语言模型时，才会这样保留前面batch的语言模型

        loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))  # batch_size * target_class_dim, batch_size
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if i % 3 == 0:
            print('epoch', epoch, 'iteration', i, 'loss', loss.item())
        if i % 5 == 0:
            val_loss = evaluate(model, val_iter)
            print('epoch', epoch, 'iteration', i, 'perplexity', np.exp(val_loss))
            if len(val_losses) == 0 or val_loss < min(val_losses):
                torch.save(model.state_dict(), 'lm.pth')
                print('best model saved to lm.pth')
            else:
                # 每次loss不降，降低学习率
                print('learning rate decay')
                scheduler.step()
            val_losses.append(val_loss)

# 读取最好模型
best_model = RNNModel(vocab_size=VOCAB_SIZE,
                      embeding_size=EMBEDDING_SIZE,
                      hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    model = model.to(device)
best_model.load_state_dict(torch.load('lm.pth'))

# 使用训练好的模型生成一些句子
hidden = best_model.init_hidden(1)
input = torch.randint(VOCAB_SIZE, (1, 1), dtype=torch.long).to(device)  # 随机拿一个数字，作为开头
words = []
for i in range(100):
    output, hidden = best_model(input, hidden)
    word_weights = output.squeeze().exp.cpu()  # squeeze：去掉所有纬度为1的纬度
    word_idx = torch.multinomial(word_weights, 1)  # 也可以用 greedy (argmax)
    input.fill_(word_idx)
    word = TEXT.vocab.itos[word_idx]
    words.append(word)
print(' '.join(words))
