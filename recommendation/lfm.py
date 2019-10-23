#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import Counter

import numpy as np
import pandas as pd


class LFM(object):
    def __init__(self, traindata, lclass=10, iters=200, alpha=0.01, lamda=0.02, topk=10, ratio=1):
        self.traindata = traindata
        self.lclass = lclass  # 隐类数量
        self.iters = iters  # 迭代次数
        self.alpha = alpha  # 梯度下降步长
        self.lamda = lamda  # 正则化参数
        self.topk = topk  # 推荐top k项
        self.ratio = ratio  # 正负样例比率
        self.user_ids_set = self._get_user_ids_set()
        self.item_ids_set = self._get_item_ids_set()
        self.item_ids_counter = self._get_item_ids_counter()

    def _get_user_ids_set(self):
        train_data = self.traindata.copy()
        user_ids = train_data['user_id'].values
        user_ids_set = set(user_ids)
        return user_ids_set

    def _get_item_ids_set(self):
        train_data = self.traindata.copy()
        item_ids = train_data['item_id'].values
        item_ids_set = set(item_ids)
        return item_ids_set

    def _get_item_ids_counter(self):
        train_data = self.traindata.copy()
        item_ids_counter = Counter(train_data['item_id'])
        return item_ids_counter

    def _get_user_positive_item(self, user_id):
        # 正样本
        train_data = self.traindata.copy()
        positive_item = train_data[train_data['user_id'] == user_id]['item_id'].values
        return positive_item

    def _get_user_negative_item(self, user_id):
        # 负样本
        train_data = self.traindata.copy()

        user_item = train_data[train_data['user_id'] == user_id]['item_id'].values
        user_item_set = set(user_item)
        user_item_length = len(user_item_set)

        negative_item_count = int(self.ratio * user_item_length)
        negative_item = list()
        for item_id, _ in self.item_ids_counter.items():
            if negative_item_count <= 0:
                break

            if item_id in user_item_set:
                continue

            negative_item.append(item_id)
            negative_item_count -= 1

        return negative_item

    def init_user_item(self, user_id):
        positive_item = self._get_user_positive_item(user_id)
        negative_item = self._get_user_negative_item(user_id)

        user_item_dict = dict()

        for item_id in positive_item:
            user_item_dict[item_id] = 1

        for item_id in negative_item:
            user_item_dict[item_id] = 0

        return user_item_dict

    def init_model(self):
        user_ids_length = len(self.user_ids_set)
        item_ids_length = len(self.item_ids_set)

        array_p = np.random.rand(user_ids_length, self.lclass)
        array_q = np.random.rand(self.lclass, item_ids_length)
        p = pd.DataFrame(array_p, columns=range(self.lclass), index=list(self.user_ids_set))
        q = pd.DataFrame(array_q, columns=list(self.item_ids_set), index=range(self.lclass))

        user_item = list()
        for user_id in self.user_ids_set:
            user_item_dict = self.init_user_item(user_id)
            user_item.append({user_id: user_item_dict})

        return p, q, user_item

    @staticmethod
    def sigmod(x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def lfm_predict(self, p, q, user_id, item_id):
        # 计算 user_id -> item_id的喜好
        p_mat = np.mat(p.ix[user_id].values)
        q_mat = np.mat(q[item_id].values).T

        pred_ui = np.dot(p_mat, q_mat).sum()
        pred_ui = self.sigmod(pred_ui)
        return pred_ui

    def lfm(self):
        p, q, user_item = self.init_model()

        for step in range(self.iters):
            for user in user_item:
                for user_id, item_ids in user.items():
                    for item_id, rui in item_ids.items():
                        eui = rui - self.lfm_predict(p, q, user_id, item_id)

                        for f_idx in range(self.lclass):
                            p[f_idx][user_id] += self.alpha * (eui * q[item_id][f_idx] - self.lamda * p[f_idx][user_id])
                            q[item_id][f_idx] += self.alpha * (eui * p[f_idx][user_id] - self.lamda * q[item_id][f_idx])

            self.alpha *= 0.9

        return p, q

    def recommend(self, p, q, user_id):
        prediction_list = list()

        for item_id in self.item_ids_set:
            predict_score = self.lfm_predict(p, q, user_id, item_id)
            prediction_list.append(predict_score)

        prediction_list.sort(reverse=True)
        return prediction_list[:self.topk]


def lfm_demo(**kwargs):
    data_path = os.path.join('data', 'ratings.dat')
    columns = ['user_id', 'item_id', 'rating', 'timestamp']

    df = pd.read_table(data_path, sep='::', names=columns, header=-1, engine='python')
    df = df.drop(columns=['rating', 'timestamp'])

    lfm = LFM(df, **kwargs)
    p_array, q_array = lfm.lfm()

    recommend = lfm.recommend(p_array, q_array, 1)
    print(recommend)


if __name__ == '__main__':
    lfm_demo(topk=10, iters=100)
