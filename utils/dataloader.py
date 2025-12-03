import numpy as np
import pandas as pd
import scipy.sparse as sp
import tqdm
import torch
from torch.utils.data import TensorDataset
import torch.utils.data as data
from sklearn.model_selection import train_test_split


def load_all(train_path, test_path, user_num, item_num):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        train_path,
        sep=',', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    # load ratings as a csr matrix
    user_list = train_data['user'].tolist()
    item_list = train_data['item'].tolist()
    pos_list = np.ones(len(item_list), dtype=float).tolist()
    train_mat = sp.csr_matrix((pos_list, (user_list, item_list)), shape=(user_num, item_num), dtype=np.float32)

    train_data = train_data.values.tolist()

    test_data = []
    # read test data, format is (3, 3) 5 7... --> (3, 3) is pos sample, follow the negative samples
    with open(test_path, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, train_mat


def cross_domain_loader(data_root, config, args, train=True):
    train_cross_loader = None
    valid_cross_loader = None
    target_user_num = config['cross_domain_targets'][args.task]['uid_'+args.target_domain]
    target_item_num =config['cross_domain_targets'][args.task]['iid_'+args.target_domain]
    train_adapt_file = data_root + '_train_cross.csv'
    test_file = data_root + '_test.negative.txt'
    print('***** data loading... *****')
    if train:
        # stage 1 training data: create train data for train cross-domain model
        co_user_num = config['cross_domain_targets'][args.task]['co_user_num']
        # X = torch.tensor(np.array(range(co_user_num)), dtype=torch.long)
        # y = torch.tensor(np.array(range(co_user_num)), dtype=torch.long)
        # train_cross_loader = TensorDataset(X, y)
        cross_data = torch.tensor(np.array(range(co_user_num)), dtype=torch.long)
        train_cross, valid_cross = train_test_split(cross_data, test_size=0.1)
        train_cross_loader = TensorDataset(train_cross,
                                           torch.tensor(np.array(range(train_cross.shape[0])), dtype=torch.long))
        valid_cross_loader = TensorDataset(valid_cross,
                                           torch.tensor(np.array(range(valid_cross.shape[0])), dtype=torch.long))
    # stage 2 training data: create train data for adapt module
    target_item_num = config['cross_domain_targets'][args.task]['iid_'+args.target_domain]
    train_adapt_data, test_data, train_mat = load_all(train_adapt_file, test_file, target_user_num, target_item_num)
    train_adapt_loader = BPRData(train_adapt_data, target_item_num, train_mat, args.train_num_ng, True)
    test_loader = BPRData(test_data, target_item_num, train_mat, 0, False)

    # return train_cross_loader, train_adapt_loader, test_loader
    return train_cross_loader, valid_cross_loader, train_adapt_loader, test_loader


class BPRData(data.Dataset):
    def __init__(self, features,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        print('negative sampling for training data')
        # self.features is the positive pairs
        for x in tqdm.tqdm(self.features, smoothing=0, mininterval=1.0):
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while self.train_mat[u, j] == 1.0:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
            self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if \
            self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
            self.is_training else features[idx][1]
        return user, item_i, item_j


