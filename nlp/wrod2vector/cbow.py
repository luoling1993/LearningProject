#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import Counter
from collections import deque

import numpy as np
import tensorflow as tf

from nlp.wrod2vector.load_data import DataLoader


class CBOW(object):
    def __init__(self, sentences=None, min_count=5, size=100, window=5,
                 max_iter=10000, negative=100, batch_size=200, learing_rate=1.0):

        if sentences is None:
            self.sentences = DataLoader().load_data()
        else:
            self.sentences = sentences

        self.min_count = min_count
        self.size = size
        self.window = window
        self.max_iter = max_iter
        self.negative = negative
        self.batch_size = batch_size
        self.learing_rate = learing_rate
        self.word_ids, self.word_ids_dict = self.build_dataset()

    def build_dataset(self):
        # 将词组映射为id
        word_counter = Counter(self.sentences)
        word_ids_dict = dict()
        word_ids_dict['unk'] = -1
        ids_index = 0
        for word, counter in word_counter.most_common():
            if counter < self.min_count:
                break
            word_ids_dict[word] = ids_index
            ids_index += 1

        # 通过映射字典将语料转为id
        word_ids_list = list()
        for word in self.sentences:
            ids = word_ids_dict.get(word, -1)
            word_ids_list.append(ids)

        return word_ids_list, word_ids_dict

    def generate_batch(self):
        length = len(self.word_ids)
        data_index = 0
        span = 2 * self.window + 1
        batch = np.ndarray(shape=(self.batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        buffer = deque(maxlen=span)

        for _ in range(span):
            buffer.append(self.word_ids[data_index])

        for idx in range(self.batch_size):
            col_idx = 0
            for span_idx in range(span):
                if span_idx == span / 2:
                    continue

                batch[span_idx, col_idx] = buffer[span_idx]
                col_idx += 1

            labels[idx, 0] = buffer[self.window]
            buffer.append(self.word_ids[data_index])
            data_index = (data_index + 1) % length

        return batch, labels

    def _train(self):
        graph = tf.Graph()

        with graph.as_default(), tf.device('/cpu:0'):
            train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, 2 * self.window], name='train_dataset')
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='train_labels')

            vocabulary_size = len(self.word_ids)
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, self.size], -1.0, 1.0), name='embeddings')

            stddev = 1.0 / np.sqrt(self.size)
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, self.size], stddev=stddev),
                                          name='softmax_wreights')

            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]), name='softmax_biases')

            context_embeddings = list()
            for idx in range(2 * self.window):
                context_embeddings.append(tf.nn.embedding_lookup(embeddings, train_dataset[:, idx]))

            avg_embed = tf.reduce_mean(tf.stack(axis=0, values=context_embeddings), 0, keepdims=False)
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases,
                                                             inputs=avg_embed, labels=train_labels,
                                                             num_sampled=self.negative, num_classes=vocabulary_size),
                                  name='loss')

            optimizer = tf.train.AdagradOptimizer(self.learing_rate).minimize(loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()

            average_loss = 0.0
            for step in range(self.max_iter):
                batch_data, batch_labels = self.generate_batch()
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}

                _, _loss = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += _loss

                if step % 1000 == 0:
                    if step > 0:
                        average_loss = average_loss / 1000

                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0.0

            final_embeddings = normalized_embeddings.eval()

        return final_embeddings

    def save_models(self, word_embedding, model_name=None):
        id_word_dict = dict(zip(self.word_ids_dict.values(), self.word_ids_dict.keys()))

        if model_name is None:
            model_name = 'cbow.bin'

        model_name = os.path.join('models', model_name)

        with open(model_name, 'w+') as f:
            for idx, item in enumerate(word_embedding):
                word = id_word_dict[idx]
                vector = ','.join([str(vec) for vec in item])
                f.write(word + '\t' + vector + '\n')

    def train(self, save=True):
        word_embedding = self._train()

        if save:
            self.save_models(word_embedding)

        return word_embedding


if __name__ == '__main__':
    cbow = CBOW()
    cbow.train()
