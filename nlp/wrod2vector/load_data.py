#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


class DataLoader(object):
    def __init__(self, corpus_name=None):
        if corpus_name is None:
            corpus_name = 'corpus.txt'
        self.corpus_path = os.path.join('data', corpus_name)

    def load_data(self):
        word_dataset = list()

        with open(self.corpus_path, encoding='utf-8') as f:
            line = f.readline()

            while line:
                line = line.strip().split(',')[1]
                words = line.split()

                word_list = list()
                for word in words:
                    if len(word) > 11:
                        continue
                    if 'nbsp' in word:
                        continue
                    word_list.append(word)

                if word_list:
                    word_dataset.append(word_list)

                line = f.readline()

        return word_dataset

DataLoader().load_data()

