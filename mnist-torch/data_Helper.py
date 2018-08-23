import torch

import pickle
import numpy as np


class dataHelper(torch.utils.data.Dataset):
    """
    My data helper
    """
    def __init__(self, input_file_path, trans, npz_path='train.dump', sub=False,
                 mode=0):
        print("I am coming")
        self.mode = mode
        try:
            self.x_data = np.array(pickle.load(open(npz_path, 'rb')))
            self.y_data = np.array(pickle.load(open(npz_path.replace('train','test'),'rb')))
            self.length = len(self.y_data)
        except:
            content = open(input_file_path, 'r').read().splitlines()[1:]
            self.x_data = [[int(t) for t in w.split(',')[1:]] for w in content]
            self.y_data = [int(w.split(',')[0]) for w in content]
            self.length = len(self.y_data)
            pickle.dump(self.x_data, open(npz_path, 'wb'))
            pickle.dump(self.y_data, open(npz_path.replace('train','test'), 'wb'))
        if mode == 0:
            all_l = self.length//10
            self.x_data = self.x_data[all_l:]
            self.y_data = self.y_data[all_l:]
            self.length = self.length - all_l
        elif mode == 1:
            all_l = self.length//10
            self.x_data = self.x_data[:all_l]
            self.y_data = self.y_data[:all_l]
            self.length = all_l
        else:
            content = open(input_file_path.replace('train', 'test'), 'r').read().splitlines()[1:]
            self.x_data = np.array([[int(t) for t in w.split(',')] for w in content])
            self.length = len(self.x_data)
        self.trans = trans

    def __getitem__(self, index):
        #return self.trans(self.x_data[index]),self.trans(self.y_data[index])
        if self.mode == 2:
            return self.x_data[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length
