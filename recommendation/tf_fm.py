#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
import tensorflow as tf


class FM(object):
    def __init__(self, input_dim, latent_k=40, learing_rate=0.005, batch_size=200, epochs=5, reg_l1=0.01, reg_l2=0.01):
        self.input_dim = input_dim
        self.latent_k = latent_k
        self.learing_rate = learing_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2

    def build_graph(self):
        data_x = tf.sparse_placeholder('float32', shape=[None, self.input_dim], name='data_x')
        data_y = tf.placeholder('int64', shape=[None, ], name='data_y')
        keep_prob = tf.placeholder('float32', name='keep_prob')

        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2], initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.input_dim, 2],
                                 initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))
            linear_terms = tf.add(tf.sparse_tensor_dense_matmul(data_x, w1), b)

        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable('v', shape=[self.input_dim, self.latent_k],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            interaction_terms = tf.multiply(0.5, tf.reduce_mean(tf.subtract(
                tf.pow(tf.sparse_tensor_dense_matmul(data_x, v), 2),
                tf.sparse_tensor_dense_matmul(data_x, tf.pow(v, 2))),
                1, keep_dims=True))

        y_out = tf.add(linear_terms, interaction_terms)
        y_out_prob = tf.nn.softmax(y_out)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=data_y, logits=y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', mean_loss)

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.FtrlOptimizer(self.learing_rate, l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss=mean_loss, global_step=global_step)



def fm_demo():
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    train_path = os.path.join('data', 'us.base')
    test_path = os.path.join('data', 'ua.test')

    train_data = pd.read_csv(train_path, sep='\t', header=-1, names=columns)
    test_data = pd.read_csv(test_path, sep='\t', header=-1, names=columns)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
