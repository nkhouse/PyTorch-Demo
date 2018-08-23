import torch
import torch.nn as nn
import torch.utils.data as TorchData
import torch.nn.functional as F
from torch.autograd import Variable
from data_Helper import dataHelper
import pandas as pd

import os

dataPath = '../data/ham'


class TextCNN(nn.Module):
    """
    My text cnn classification
    """
    def __init__(self, vocab_size, max_length, filter_sizes=[3, 4, 5],
                 num_filters=100, emb_size=100, outpu_size=2):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.emb_size = emb_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filters,
                    kernel_size=[filter_sizes[0], emb_size], stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[max_length-filter_sizes[0]+1, 1])
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filters,
                      kernel_size=[filter_sizes[1], emb_size], stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=[max_length-filter_sizes[1]+1, 1])
        )
        self.out = nn.Linear(num_filters * 2, outpu_size)

    def forward(self, input):
        input = self.embedding(input)
        input = input.view(-1, 1, self.max_length, self.emb_size)
        out1 = self.conv1(input).view(-1, self.num_filters)
        out2 = self.conv2(input).view(-1, self.num_filters)
        out = torch.cat([out1, out2], 1)
        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out


def train_valid_predict():
    train_helper = dataHelper(os.path.join(dataPath, 'train.csv'), mode=0)
    valid_helper = dataHelper(os.path.join(dataPath, 'train.csv'), mode=1)
    test_helper = dataHelper(os.path.join(dataPath, 'train.csv'),mode=2)

    trainLoader = TorchData.DataLoader(dataset=train_helper, batch_size=100,
                                       shuffle=True, num_workers=2)
    validLoader = TorchData.DataLoader(dataset=valid_helper, batch_size=100,
                                       shuffle=True, num_workers=2)
    testLoader = TorchData.DataLoader(dataset=test_helper, batch_size=100,
                                       shuffle=False, num_workers=2)
    max_length = train_helper.max_length
    vocab_size = train_helper.vocab_size

    model = TextCNN(vocab_size, max_length)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ceriation = nn.CrossEntropyLoss()

    for j in range(100):
        for i, (x_data, y_data) in enumerate(trainLoader):
            optimizer.zero_grad()
            x, target = Variable(x_data.cuda()), Variable(y_data.cuda())
            out = model(x)
            loss = ceriation(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and i > 0:
                print("Step: %d, loss %.4f" % (i, loss))
        total_cnt = 0
        correct_cnt = 0
        for i, (x_data, y_data) in enumerate(validLoader):
            x, target = Variable(x_data.cuda()), Variable(y_data.cuda())
            out = model(x)
            loss = ceriation(out, target)
            _, pre_label = torch.max(out.data,1)
            total_cnt += x.data.size(0)
            correct_cnt += (pre_label == target.data).sum()
        print("Test step: %d, loss %.4f, acc %.4f" % (j, loss, correct_cnt.cpu().numpy()/total_cnt))

    result = []
    smsId= []
    for i, (sm, x) in enumerate(testLoader):
        x = Variable(x.cuda())
        out = model(x)
        _, pre_label = torch.max(out.data, 1)
        result += [w for w in pre_label.cpu().numpy()]
        smsId += [w for w in sm.cpu().numpy()]
    result = ['ham' if w == 0 else 'spam' for w in result]
    res = pd.DataFrame({"SmsId": smsId, 'Label':result})
    res.to_csv(os.path.join(dataPath, "sub.csv"), index=False, columns=list(res))

if __name__ == '__main__':
    train_valid_predict()

