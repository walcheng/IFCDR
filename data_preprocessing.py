"""
This py generates four types of data file, _test.csv for test in single or cross-domain,
_train_single.csv is the training set for single domain training, _train_cross is the
training set for cross-model
"""
import numpy as np
import random
import argparse
import json
import scipy.sparse as sp
from utils.preprocessing import DataPreprocessingMid, DataPreprocessingReady
import pandas as pd
import tqdm


def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=1)
    parser.add_argument('--task', default='1')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--test_neg_sample', default=True, help='generate negative samples for each domains')
    parser.add_argument('--neg_num', type=int, default=99, help='test data negative sample')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['task'] = args.task
    return args, config


if __name__ == '__main__':
    config_path = 'config.json'
    args, config = prepare(config_path)

    if args.process_data_mid:
        for dealing in ['Books', 'Electronics', 'Video_Games']:
            DataPreprocessingMid(config['root'], dealing).main()
    if args.process_data_ready:
        for task in ['1']:
            DataPreprocessingReady(config['root'], config['cross_domain_targets'], task, config['ratio']).main()
            print('task:{};seed:{};'.format(args.task, args.seed))
    # create negative samples for bpr loss training
    if args.test_neg_sample:
        cross_dom = ['idom', 'jdom', 'kdom']
        for task in [args.task]:
            data_root = config['root'] + 'ready/' + str(int(config['ratio'][0] * 10)) + '_' +\
                        str(int(config['ratio'][1] * 10)) + '/task_' + task
            for dom in cross_dom:
                # merge train and test data into overall rating matrix
                neg_sample_dom = config['cross_domain_targets'][task][dom]
                train_path = data_root + '/' + neg_sample_dom + '_train_single.csv'
                test_path = data_root + '/' + neg_sample_dom + '_test.csv'
                df_train_data = pd.read_csv(train_path, usecols=[0, 1], header=None)
                df_test_data = pd.read_csv(test_path, usecols=[0, 1], header=None)
                df_data = pd.concat([df_train_data, df_test_data], axis=0, ignore_index=True)
                user_num = df_data[0].max() + 1
                item_num = df_data[1].max() + 1
                list_test = df_test_data.values.tolist()

                # load ratings as a dok matrix
                user_list = df_data[0].tolist()
                item_list = df_data[1].tolist()
                pos_list = np.ones(len(item_list), dtype=float).tolist()
                train_mat = sp.csr_matrix((pos_list, (user_list, item_list)), shape=(user_num, item_num),
                                          dtype=np.float32)

                # create negative sample into test data
                with open(data_root+'/' + neg_sample_dom + '_test.negative.txt', "a") as f:
                    print('negative sampling for domain {}....'.format(neg_sample_dom))
                    for pos_pair in tqdm.tqdm(list_test, smoothing=0, mininterval=1.0):
                        u, i = pos_pair[0], pos_pair[1]
                        list_line = [str((u, i))]
                        count = args.neg_num
                        while count > 0:
                            j = np.random.randint(item_num)
                            while train_mat[u, j] == 1.0:
                                j = np.random.randint(item_num)
                            list_line.append(str(j))
                            count -= 1
                        line = '\t'.join(list_line)
                        f.write(line + '\n')
                    f.close()








