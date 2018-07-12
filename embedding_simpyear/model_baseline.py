from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics as mt
import scipy.stats as stats

import collections
import math
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from six.moves import xrange

import datum

data_index = 0
len_stock = 3145
len_fund  = 2199


class Model():
    def __init__(self, learning_rate_rank):
        self.save_rank = learning_rate_rank
        self.learning_rate = 1 / np.power(10, learning_rate_rank)
        self.batch_size = 10

    def data_initial(self, datum = None):
        if datum is not None:
            self.data = datum
        else:
            self.data = datum.Datum()
            self.data.data_prepare()
            self.data.evaluation_prepare()
            self.data.label_trend()
            self.data.label_return()
        
    def data_split(self):
        self.day_sample = self.data.price_data.shape[1]
        self.stock_sample = self.data.price_data.shape[0]
        
        self.train_vali = self.day_sample // 2
        self.vali_test = self.train_vali + self.day_sample // 4
        
        self.rank_day = np.array(range(self.train_vali))
        self.rank_stock = np.array(range(self.stock_sample))
        
    def factor_network(self):
        learning_rate = self.learning_rate
        
        self.embedding = tf.placeholder(tf.float32, shape=[32, None], name='embedding')
        self.factor = tf.placeholder(tf.float32, shape=[44, None], name='factor')
        self.factor_index = tf.placeholder(tf.int32, shape=[5], name='factor_index')
        self.label = tf.placeholder(tf.int32, shape=[None, 2], name='lable')
        
        self.weight = tf.get_variable(name='weight', shape=[76, 44], initializer=tf.truncated_normal_initializer(stddev=1.0))
        self.bias = tf.get_variable(name='bias', shape=[44], initializer=tf.zeros_initializer)
        
        self.weight_index = tf.concat([self.factor_index, tf.constant([i for i in range(44, 76)])], axis=0)
        self.weight_select = tf.nn.embedding_lookup(self.weight, self.weight_index)
        self.factor_select = tf.nn.embedding_lookup(self.factor, self.factor_index)
        
        self.part_input = tf.concat([self.factor_select, self.embedding], axis=0)
        self.part_input = tf.transpose(self.part_input)
        self.hidden = tf.matmul(self.part_input, self.weight_select) + self.bias 
        self.hidden_select = tf.nn.embedding_lookup(tf.transpose(self.hidden), self.factor_index)
        
        all_feature = tf.concat([tf.transpose(self.factor_select), tf.transpose(self.hidden_select)], 1)
        self.pred = tf.contrib.layers.fully_connected(
            inputs=all_feature,
            num_outputs=2,  # hidden
            activation_fn=tf.tanh,
            weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
            biases_initializer=tf.zeros_initializer()
        )
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.label))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        
        self.all_input = tf.concat([self.factor, self.embedding], axis=0)
        self.all_input = tf.transpose(self.all_input)
        self.evaluation = tf.matmul(self.all_input, self.weight) + self.bias
        
        saver = tf.train.Saver(max_to_keep=5)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.save(sess, 'baselineModel_initial/logmodel.ckpt')

    def training(self):
        epochs = 50
        batch_num = self.train_vali // self.batch_size
        saver = tf.train.Saver(max_to_keep=None)
        with tf.Session() as sess:
            saver.restore(sess, 'baselineModel_initial/logmodel.ckpt')
            for epoch in range(epochs):
                print('epoch: {}'.format(epoch))
                # validation
                new_f_total = []
                use_index = []
                for stock_idx in range(self.stock_sample):
                    price = self.data.price_data[stock_idx, self.train_vali:self.vali_test]
                    feature = self.data.feature_data[stock_idx, self.train_vali:self.vali_test, :]
                    label = self.data.ar_trend[stock_idx, self.train_vali:self.vali_test, :]
                    stock_name = str(self.data.code_tag[stock_idx])
                    for _ in range(6-len(stock_name)):
                        stock_name = '0' + stock_name
                    if stock_name not in self.data.list_stocks:
                        continue
                    else:
                        use_index.append(stock_idx)
                        embed = np.expand_dims(self.data.embedding[self.data.list_stocks.index(stock_name)], axis=0)
                    embed = np.repeat(embed, self.vali_test-self.train_vali, 0)
                    embed = embed.T
                    factor_index = random.sample(list(range(44)), 5)
                    feed_dict = {self.embedding: embed, self.factor: feature.T, self.factor_index: factor_index, self.label: label}
                    new_f= sess.run(self.evaluation, feed_dict=feed_dict)        
                    new_f_total.append(new_f)
                new_f_total = np.array(new_f_total)
                use_index = np.array(use_index)
                ic = np.zeros((44, self.vali_test-self.train_vali, 4))
                for day in range(0, self.vali_test-self.train_vali):
                    for fac in range(44):
                        for id_idx in range(4):
                            rank_ic = stats.spearmanr(self.data.ar_ic[use_index, day, id_idx], new_f_total[:, day, fac])
                            ic[fac, day, id_idx] = rank_ic[0]
                ic = ic.mean(axis=1)
                print(ic)
                if not os.path.exists('baselineModel-'+str(self.save_rank)):
                    os.mkdir('baselineModel-'+str(self.save_rank))
                    os.mkdir('data/baselineEvaluation-'+str(self.save_rank))
                pd.DataFrame(ic).to_csv('data/baselineEvaluation-{}/epoch_evaluation_{}.csv'.format(self.save_rank, epoch))
                saver.save(sess, 'baselineModel-{}/logmodel.ckpt'.format(self.save_rank), global_step=epoch)   
                
                np.random.shuffle(self.rank_day)
                loss_all = 0
                loss_count = 0
                for batch_count in range(batch_num):#batch_num
                    print('batch_count:{}'.format(batch_count))
                    np.random.shuffle(self.rank_stock)
                    for stock_count, stock_idx in enumerate(self.rank_stock):
                        price = self.data.price_data[stock_idx, self.rank_day[batch_count*self.batch_size:(batch_count+1)*self.batch_size]]
                        feature = self.data.feature_data[stock_idx, self.rank_day[batch_count*self.batch_size:(batch_count+1)*self.batch_size], :]
                        label = self.data.ar_trend[stock_idx, self.rank_day[batch_count*self.batch_size:(batch_count+1)*self.batch_size], :]
                        stock_name = str(self.data.code_tag[stock_idx])
                        for _ in range(6-len(stock_name)):
                            stock_name = '0' + stock_name
                        if stock_name not in self.data.list_stocks:
                            continue
                        else:
                            embed = np.expand_dims(self.data.embedding[self.data.list_stocks.index(stock_name)], axis=0)
                        embed = np.repeat(embed, self.batch_size, 0)
                        embed = embed.T
                        for epoch_factor in range(30):
                            factor_index = random.sample(list(range(44)), 5)
                            feed_dict = {self.embedding: embed, self.factor: feature.T, self.factor_index: factor_index, self.label: label}
                            _, loss_val = sess.run([self.optimizer, self.cost], feed_dict=feed_dict)
                            loss_all += loss_val
                            loss_count += 1
                    print('avg_loss: {}'.format(loss_all/loss_count))
                    loss_all = 0
                    loss_count = 0
                
            







