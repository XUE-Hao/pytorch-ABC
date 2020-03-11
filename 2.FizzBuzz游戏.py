import numpy as np
import torch


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decoder(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


def helper(i):
    print(fizz_buzz_decoder(i, fizz_buzz_encode(i)))


# for i in range(1, 16):
#     helper(i)

NUM_DIGITS = 10


def binary_encoder(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


# print(binary_encoder(31, NUM_DIGITS))

trX = torch.Tensor([binary_encoder(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# print([binary_encoder(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
print(trY)

NUM_HIDDEN = 100

model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)  # softmax后，得到一个概率分部
)
if torch.cuda.is_available():
    model = model.cuda()

loss_fn = torch.nn.CrossEntropyLoss()  # 拟合两种分部的相似度
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
BATCH_SIZE = 128

for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        if torch.cuda.is_available():
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)  # CrossEntropyLoss参数input是一个概率分部,taget是类别标签
        print('Epoch', epoch, loss.item())

        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

testX = torch.Tensor([binary_encoder(i, NUM_DIGITS) for i in range(1, 101)])
if torch.cuda.is_available():
    testX = testX.cuda()
with torch.no_grad():  # 测试时，不要梯度下降
    testY = model(testX)

predictions = zip(range(1, 101), testY.max(1)[1].cpu().tolist())  # 取概率最大值，并且和输入压缩
print([fizz_buzz_decoder(i, x) for i, x in predictions])
