#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def insert_sorted(lists):
    """
    插入排序
    - 在第i轮通过列表时(i从1到n-1),第i项应该插入到列表的前i个项中的正确位置
    - 在第i轮之后，列表的前i项应该是排好序的
    :param lists: 乱序list
    :return: 顺序list
    """
    lists = lists.copy()

    length = len(lists)
    for i in range(1, length):
        value = lists[i]
        j = i - 1
        while j >= 0:
            if lists[j] > value:
                lists[j + 1] = lists[j]
                lists[j] = value

            j -= 1

    return lists


def test_sorted():
    origin_list = np.random.randint(0, 10, size=(1, 10)).ravel()
    print(origin_list)
    sorted_list = insert_sorted(origin_list)
    print(sorted_list)


if __name__ == '__main__':
    test_sorted()
