import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import random
import spacy

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)  # 表示句子是哪个类型

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# 查看每个数据split有多少条数据
# print(f'number of training examples: {len(train_data)}')
# print(f'number of testing examples: {len(test_data)}')
# 查看一个example
# print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(random_state=random.seed(SEED))
# print(f'number of training examples: {len(train_data)}')
# print(f'number of validation examples: {len(valid_data)}')
# print(f'number of testing examples: {len(test_data)}')

# TEXT.build_vocab(train_data, max_size=25000)
# LABEL.build_vocab(train_data)
# vector='glove.6B.100d' 使用预训练好的词向量
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)
# print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
# print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')

BATCH_SIZE = 64

device = torch.device('duda' if torch.cuda.is_available() else 'cpu')

# BucketIterator把长度差不多的句子放到同一个batch中，确保每个batch中不会出现太多padding
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)

# batch = next(iter(valid_iterator))
# print(batch.text)


class WordAvGModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx):
        super(WordAvGModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.linear = nn.Linear(embedding_size, output_size)

    def forward(self, text):
        embedded = self.embed(text)  # [seq_len, batch_size, embedding_size]
        # embedded = embedded.transpose(1, 0)  # [batch_size, seq_len, embedding_size]
        embedded = embedded.permute(1, 0, 2)  # [batch_size, seq_len, embedding_size]
        pooled = F.avg_pool2d(embedded, [embedded.shape[1], 1]).squeeze()  # [batch_size, embedding_size]
        return self.linear(pooled)


VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_SIZE = 100
OUTPUT_SIZE = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = WordAvGModel(vocab_size=VOCAB_SIZE,
                     embedding_size=EMBEDDING_SIZE,
                     output_size=OUTPUT_SIZE,
                     pad_idx=PAD_IDX)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(count_parameters(model))
# 使用预训练词向量初始化
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

# 训练模型
optimizer = torch.optim.Adam(model.parameters())
crit = nn.BCEWithLogitsLoss()  # 只针对二分类问题

model = model.to(device)
crit = crit.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))  # round: 四舍五入
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.train()
    total_len = 0.
    for batch in iterator:
        preds = model(batch.text).squeeze()  # [batch_size, 1]中的1纬度需要用squeeze压掉
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)
    return epoch_loss / total_len, epoch_acc /total_len


def evaluate(model, iterator, crit):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.
    for batch in iterator:
        preds = model(batch.text).squeeze()
        loss = crit(preds, batch.label)
        acc = binary_accuracy(preds, batch.label)

        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)
    model.train()
    return epoch_loss / total_len, epoch_acc /total_len


N_EPOCH = 10
best_valid_acc = 0.
for epoch in range(N_EPOCH):
    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'wordavg-model.pth')

    print('Epoch', epoch, 'Train Loss', train_loss, 'Train_Acc', train_acc)
    print('Epoch', epoch, 'Valid Loss', valid_loss, 'Valid_Acc', valid_acc)

model.load_state_dict(torch.load('wordavg-model.pth'))
nlp = spacy.load('en')


def predict_sentiment(sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)  # [seq_len] embedding前的词id要使用LongTensor
    tensor = tensor.unsqueeze(1)  # [seq_len * batch_size(1)]
    pred = torch.sigmoid(model(tensor))
    return pred.item()


predict_sentiment('This film is horrible!')
predict_sentiment('This film is terrific!')
predict_sentiment('This film is terrible!')


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, hidden_size, dropout):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=2)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embed(text)  # [seq_len, batch_size, embedding_size]
        embedded = self.dropout(embedded)  # dropout一般用在embedding后
        output, (hidden, cell) = self.lstm(embedded)  # lstm可以在传一个hidden，不传为全0
        # hiddden: 2 * batch_size * hidden_size
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)  # 把两个hidden拼到一起
        hidden = self.dropout(hidden.squeeze())
        return self.linear(hidden)


model = RNNModel(vocab_size=VOCAB_SIZE,
                 embedding_size=EMBEDDING_SIZE,
                 output_size=OUTPUT_SIZE,
                 pad_idx=PAD_IDX,
                 hidden_size=100,
                 dropout=0.5)
# 使用预训练词向量初始化
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)
model = model.to(device)

N_EPOCH = 10
best_valid_acc = 0.
for epoch in range(N_EPOCH):
    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'lstm-model.pth')

    print('Epoch', epoch, 'Train Loss', train_loss, 'Train_Acc', train_acc)
    print('Epoch', epoch, 'Valid Loss', valid_loss, 'Valid_Acc', valid_acc)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, pad_idx, num_filters, filter_sizes, dropout):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters,
                      kernel_size=(fs, embedding_size))
            for fs in filter_sizes
        ])  # 3个CNN
        # self.cov = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(filter_size, embedding_size))
        self.linear = nn.Linear(embedding_size * len(filter_sizes), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)  # [batch_size, seq_len]
        embedded = self.embed(text)  # [batch_size, seq_len, embedding_size]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_size]
        # conved = F.relu(self.conv(embedded))  # [batch_size, num_filters, seq_len-filter_size+1, 1]
        # conved = conved.squeeze()  # [batch_size, num_filters, seq_len-filter_size+1]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # max over time polling
        # pooled = F.max_pool1d(conved, conved.shape[2])  # [batch_size, num_filters, 1]
        # pooled = pooled.squeeze(2)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        pooled = torch.cat(pooled, dim=1)  # batch_size, 3 * num_filters
        pooled = self.dropout(pooled)
        return self.linear(pooled)


model = CNN(vocab_size=VOCAB_SIZE,
            embedding_size=EMBEDDING_SIZE,
            output_size=OUTPUT_SIZE,
            pad_idx=PAD_IDX,
            num_filters=100,
            filter_sizes=[3, 4, 5],
            dropout=0.5)
pretrained_embedding = TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)
model = model.to(device)

N_EPOCH = 10
best_valid_acc = 0.
for epoch in range(N_EPOCH):
    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), 'cnn-model.pth')

    print('Epoch', epoch, 'Train Loss', train_loss, 'Train_Acc', train_acc)
    print('Epoch', epoch, 'Valid Loss', valid_loss, 'Valid_Acc', valid_acc)

model.load_state_dict(torch.load('cnn-model.pth'))
test_loss, test_acc = evaluate(model, test_iterator, crit)
print('CNN model test loss: ', test_loss, 'accuracy:', test_acc)
