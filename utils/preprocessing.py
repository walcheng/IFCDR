import pandas as pd
import gzip
import json
import tqdm
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split


class DataPreprocessingMid():
    def __init__(self,
                 root,
                 dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        print('Parsing ' + self.dealing + ' Mid...')
        re = []
        with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
                re.append([line['reviewerID'], line['asin'], line['overall']])
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
        print(self.dealing + ' Mid Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=False)
        return re


class DataPreprocessingReady():
    def __init__(self,
                 root,
                 cross_domain_targets,
                 task,
                 ratio):
        self.root = root
        self.idom = cross_domain_targets[task]['idom']
        self.jdom = cross_domain_targets[task]['jdom']
        self.kdom = cross_domain_targets[task]['kdom']
        self.ratio = ratio
        self.task = task

    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        if 'uid' in re.columns:
            return re
        else:
            re.columns = ['uid', 'iid', 'y', 'timestamp']
            return re


    def mapper(self, idom, jdom, kdom):
        print('idom inters: {}, uid: {}, iid: {}.'.format(len(idom), len(set(idom.uid)), len(set(idom.iid))))
        print('jdom inters: {}, uid: {}, iid: {}.'.format(len(jdom), len(set(jdom.uid)), len(set(jdom.iid))))
        print('kdom inters: {}, uid: {}, iid: {}.'.format(len(kdom), len(set(kdom.uid)), len(set(kdom.iid))))
        set_uid_idom = set(idom.uid)
        set_uid_jdom = set(jdom.uid)
        set_uid_kdom = set(kdom.uid)
        set_iid_idom = set(idom.iid)
        set_iid_jdom = set(jdom.iid)
        set_iid_kdom = set(kdom.iid)
        co_uid = set_uid_idom & set_uid_jdom & set_uid_kdom
        out_uid_idom = set_uid_idom - co_uid
        out_uid_jdom = set_uid_jdom - co_uid
        out_uid_kdom = set_uid_kdom - co_uid
        all_uid = set_uid_idom | set_uid_jdom | set_uid_kdom
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        # put co_user at the former
        uid_idom_dict = dict(zip(list(co_uid) + list(out_uid_idom), range(len(set_uid_idom))))
        uid_jdom_dict = dict(zip(list(co_uid) + list(out_uid_jdom), range(len(set_uid_jdom))))
        uid_kdom_dict = dict(zip(list(co_uid) + list(out_uid_kdom), range(len(set_uid_kdom))))
        iid_dict_idom = dict(zip(set_iid_idom, range(len(set_iid_idom))))
        iid_dict_jdom = dict(zip(set_iid_jdom, range(len(set_iid_jdom))))
        iid_dict_kdom = dict(zip(set_iid_kdom, range(len(set_iid_kdom))))
        idom.uid = idom.uid.map(uid_idom_dict)
        jdom.uid = jdom.uid.map(uid_jdom_dict)
        kdom.uid = kdom.uid.map(uid_kdom_dict)
        idom.iid = idom.iid.map(iid_dict_idom)
        jdom.iid = jdom.iid.map(iid_dict_jdom)
        kdom.iid = kdom.iid.map(iid_dict_kdom)
        return idom, jdom, kdom, len(co_uid)

    def split_co_users(self, co_users, dom):
        """
        :param co_users: overlapping user
        :param dom: domain data
        :return: training data for single domain, training data for cross-domain, test data for single and cross-domain
        """
        dom_users = set(dom.uid.unique())
        dom_co_users = dom[dom['uid'].isin(co_users)]
        # train set for cross-domain training, test for cross-domain and single domain testing
        dom_train_cross, dom_test = train_test_split(dom_co_users, test_size=self.ratio[0])
        dom_wo_co_users = dom[dom['uid'].isin(dom_users - co_users)]
        # train set for train single domain model
        dom_train_single = pd.concat([dom_train_cross, dom_wo_co_users], ignore_index=True, axis=0)
        return dom_train_single, dom_train_cross, dom_test

    def split(self, idom, jdom, kdom, co_user_num):
        print('All iid: {}.'.format(len(set(idom.iid) | set(jdom.iid) | set(kdom.iid))))
        co_users = set(list(range(co_user_num)))
        idom_train_single, idom_train_cross, idom_test = self.split_co_users(co_users, idom)
        jdom_train_single, jdom_train_cross, jdom_test = self.split_co_users(co_users, jdom)
        kdom_train_single, kdom_train_cross, kdom_test = self.split_co_users(co_users, kdom)
        return idom_train_single, idom_train_cross, idom_test, \
               jdom_train_single, jdom_train_cross, jdom_test, \
               kdom_train_single, kdom_train_cross, kdom_test

    def save(self, idom_train_single, idom_train_cross, idom_test,
             jdom_train_single, jdom_train_cross, jdom_test,
             kdom_train_single, kdom_train_cross, kdom_test):
        output_root = self.root + 'ready/' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/task_' + str(self.task)
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        idom_train_single.to_csv(output_root + '/' + self.idom + '_train_single.csv', sep=',', header=None, index=False)
        idom_train_cross.to_csv(output_root + '/' + self.idom + '_train_cross.csv', sep=',', header=None, index=False)
        idom_test.to_csv(output_root + '/' + self.idom + '_test.csv', sep=',', header=None, index=False)
        jdom_train_single.to_csv(output_root + '/' + self.jdom + '_train_single.csv', sep=',', header=None, index=False)
        jdom_train_cross.to_csv(output_root + '/' + self.jdom + '_train_cross.csv', sep=',', header=None, index=False)
        jdom_test.to_csv(output_root + '/' + self.jdom + '_test.csv', sep=',', header=None, index=False)
        kdom_train_single.to_csv(output_root + '/' + self.kdom + '_train_single.csv', sep=',', header=None, index=False)
        kdom_train_cross.to_csv(output_root + '/' + self.kdom + '_train_cross.csv', sep=',', header=None, index=False)
        kdom_test.to_csv(output_root + '/' + self.kdom + '_test.csv', sep=',', header=None, index=False)

    def main(self):
        random.seed(2020)
        np.random.seed(2020)
        idom = self.read_mid(self.idom)
        jdom = self.read_mid(self.jdom)
        kdom = self.read_mid(self.kdom)
        # For each domain, makes the co-users in the front of other users.
        idom, jdom, kdom, co_uid_num = self.mapper(idom, jdom, kdom)
        idom_train_single, idom_train_cross, idom_test, \
        jdom_train_single, jdom_train_cross, jdom_test, \
        kdom_train_single, kdom_train_cross, kdom_test = self.split(idom, jdom, kdom, co_uid_num)
        self.save(idom_train_single, idom_train_cross, idom_test,
                  jdom_train_single, jdom_train_cross, jdom_test,
                  kdom_train_single, kdom_train_cross, kdom_test)
