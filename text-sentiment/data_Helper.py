import torch
from nltk.corpus import stopwords
import re
import pickle
import numpy as np


stops = stopwords.words('english')
min_freq = 1


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def prePorcessor(content):
    content = [clean_str(w) for w in content]
    content = [w.split() for w in content]
    w2c = dict()
    for lines in content:
        for word in lines:
            if word in stops:
                continue
            if not word in w2c:
                w2c[word] = 1
            else:
                w2c[word] += 1
    w2c = sorted(w2c.items(), key=lambda x:x[1], reverse=False)
    w2c = [(w, z) for w, z in w2c if z > min_freq]
    w2i = {w:i+1 for i, (w, z) in enumerate(w2c)}
    w2i[' '] = 0
    content = [[w2i[w] for w in line if w in w2i] for line in content]
    max_length = max([len(w) for w in content])
    res = []
    for line in content:
        length = len(line)
        tmp = line + [0 for w in range(max_length - length)]
        res.append(tmp)
    return res, len(w2i)


class dataHelper(torch.utils.data.Dataset):
    """
    My data helper
    """
    def __init__(self, input_file_path, npz_path='train.dump', sub=False,
                 mode=0):
        print("I am coming")
        self.mode = mode
        try:
            self.x_data1 = pickle.load(open(npz_path, 'rb'))
            self.x_data2 = pickle.load(open(npz_path.replace('train', 'test'), 'rb'))
            self.y_data = pickle.load(open(npz_path.replace('train', 'train_y'), 'rb'))
            self.ids = pickle.dump(open('ids.dump', 'wb'))
            self.vocab_size = len(set(np.reshape(self.x_data,[1, -1])))
        except:
            content = open(input_file_path, 'r').read().splitlines()[1:]
            self.y_data = [0 if w.split(',')[0] == 'ham' else 1 for w in content]
            x_data1 = [','.join(w.split(',')[1:]) for w in content]
            content1 = open(input_file_path.replace('train', 'test'),'r').read().splitlines()[1:]
            x_data2 = [','.join(w.split(',')[1:]) for w in content1]
            self.ids = [int(w.split(',')[0]) for w in content1]
            merge, self.vocab_size = prePorcessor(x_data1+x_data2)
            self.x_data1 = merge[:len(self.y_data)]
            self.x_data2 = merge[len(self.y_data):]
            pickle.dump(self.x_data1, open(npz_path, 'wb'))
            pickle.dump(self.x_data2, open(npz_path.replace('train','test'), 'wb'))
            pickle.dump(self.y_data, open(npz_path.replace('train', 'train_y'), 'wb'))
            pickle.dump(self.ids, open('ids.dump', 'wb'))

        all_l = len(self.y_data) // 10
        if mode == 2:
            self.x_data = np.array(self.x_data2)
        elif mode == 0:
            self.x_data = np.array(self.x_data1[all_l:])
            self.y_data = np.array(self.y_data[all_l:])
        else:
            self.x_data = np.array(self.x_data1[:all_l])
            self.y_data = np.array(self.y_data[:all_l])


        self.max_length = len(self.x_data[0])

    def __getitem__(self, index):
        if self.mode == 2:
            return self.ids[index], self.x_data[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)
