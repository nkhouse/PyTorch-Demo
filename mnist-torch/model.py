import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as TorchData
from torchvision.transforms import transforms
from torch.autograd import Variable
from data_Helper import dataHelper
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime

pri_path = "../data/mnist"
start = datetime.now()


class MLP(nn.Module):
    """
    My mlp for mnist implements with pytorch
    """
    def __init__(self, input_size=784, output_size=10, hidden_size=[500,500]):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)


    def forward(self, input):
        input = input.view(-1, self.input_size).float()
        out = F.sigmoid(self.fc1(input))
        out = F.sigmoid(self.fc2(out))
        out = F.log_softmax(self.fc3(out), 1)
        return out


class CNN(nn.Module):
    """
    My cnn model for mnist implements with pytorch
    """
    def __init__(self, input_size=28, output_size=10):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(64*7*7, output_size)

    def forward(self, input):
        input = input.view(-1,1,self.input_size, self.input_size).float()
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.out(out)
        out = F.log_softmax(out, 1)
        return out


def train():
    #model = MLP()
    model = CNN()
    model.cuda()
    trans = transforms.Compose([transforms.ToTensor()])

    print(model)
    data_helper = dataHelper(os.path.join(pri_path, 'train.csv'), trans, mode=0)
    valid_helper = dataHelper(os.path.join(pri_path, 'train.csv'), trans, mode=1)
    test_helper = dataHelper(os.path.join(pri_path, 'train.csv'), trans, mode=2)

    train_loader = TorchData.DataLoader(dataset=data_helper, batch_size=100,
                                        shuffle=True, num_workers=2)
    valid_loader = TorchData.DataLoader(dataset=valid_helper, batch_size=100,
                                        shuffle=True, num_workers=2)
    test_loader = TorchData.DataLoader(dataset=test_helper, batch_size=100,
                                       shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ceriation = nn.CrossEntropyLoss()

    ave_loss = 0.0
    for j in range(100):
        for i, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            out = model(x)
            loss = ceriation(out, target)
            #ave_loss = ave_loss*0.9 + loss.data[0]*0.1
            loss.backward()
            optimizer.step()
            if i % 100 == 0 and i > 0:
                print("Step: %d, loss: %.4f" % (i, loss))
        correct_cnt, ave_loss = 0, 0.0
        total_cnt = 0
        for i, (x, target) in enumerate(valid_loader):
            x, target = Variable(x.cuda()), Variable(target.cuda())
            out = model(x)
            loss = ceriation(out, target)
            _, pre_label = torch.max(out.data,1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pre_label == target.data).sum()
        print("------------\nTest step:%d, test acc: %.4f" % (j, correct_cnt.cpu().numpy()/total_cnt))
        print('-----------------\n')
        res = []
    for i, x in enumerate(test_loader):
        x = Variable(x.cuda())
        out = model(x)
        _, pre_label = torch.max(out.data, 1)
        res += [w.cpu().numpy() for w in pre_label]
    res = pd.DataFrame({'ImageId':[w+1 for w in range(len(res))], 'Label': res})
    res.to_csv('sub.csv', index=False, columns=list(res))


if __name__ == '__main__':
    train()
    print(datetime.now() - start)
