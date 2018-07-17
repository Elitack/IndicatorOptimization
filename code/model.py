from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import linear_model as LR
from sklearn import metrics as mt
import scipy.stats as stats

import sys
import math
import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from datum import *

data_index = 0
len_stock = 3145
len_fund  = 2199

data_dir = '../data/result/'

class Model():
    def __init__(self, learning_rate_rank, param):
        self.save_rank = learning_rate_rank
        self.learning_rate = 1 / np.power(10, learning_rate_rank)
        self.batch_size = 10
        self.param = param
        self.best_epoch = -1

    def data_initial(self, datum):
        self.data = datum
        
    def data_split(self):
        self.day_sample = self.data.price_data.shape[1]
        self.stock_sample = self.data.price_data.shape[0]
        
        self.train_vali = self.day_sample // 2
        self.vali_test = self.train_vali + self.day_sample // 4
        
        self.rank_day = np.array(range(self.train_vali))

    def factor_network(self):
        learning_rate = self.learning_rate
        
        self.embedding = tf.placeholder(tf.float32, shape=[len(self.data.use_index), 32], name='embedding')
        self.factor = tf.placeholder(tf.float32, shape=[len(self.data.use_index)], name='factor')
        self.factor_index = tf.placeholder(tf.int32, shape=[1], name='factor_index')
        self.ic = tf.placeholder(tf.float32, shape=[len(self.data.use_index)], name='ic')
        
        self.u_bias = tf.get_variable('u_bias', shape=[44, 32], initializer=tf.truncated_normal_initializer(stddev=1.0))
        
        self.u_bias_select = tf.nn.embedding_lookup(self.u_bias, self.factor_index)
        
        self.hidden = tf.matmul(self.embedding, tf.transpose(self.u_bias_select))
        self.confidence = tf.exp(self.hidden) / tf.reduce_sum(tf.exp(self.hidden))
        
        self.new_f = tf.multiply(self.factor, tf.reshape(self.confidence, [-1]))
        mean, var = tf.nn.moments(self.new_f, axes=[0])
        self.process_new_f = (self.new_f - mean) / tf.sqrt(var)
        
        mean, var = tf.nn.moments(self.ic, axes=[0])
        self.process_ic = (self.ic - mean) / tf.sqrt(var)
        
        self.correlation = tf.reduce_sum(self.process_new_f * self.process_ic) / len(self.data.list_stocks)
        # self.cost = -tf.log(tf.abs(self.correlation))
        self.cost = tf.exp(-10*tf.abs(self.correlation))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        
        
    def training(self):
        epochs = 1000
        batch_num = self.train_vali // self.batch_size
        saver = tf.train.Saver(max_to_keep=None)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for fac_idx in range(44):
                print('factor: {}'.format(fac_idx))
                vali_loss_max = np.inf
                tolerance = 3
                for epoch in range(epochs):
                    loss_all = 0
                    loss_count = 1
                    embedding = self.data.embedding[self.data.use_index, :]
                    for count_day, day in enumerate(range(self.train_vali,self.vali_test)):#batch_num
                        feature = self.data.feature_data[self.data.use_index, day, fac_idx]
                        if feature.std() != 0:
                            feature = (feature-feature.min()) / (feature.max() - feature.min())
                        else:
                            continue
                        label = self.data.ar_ic[self.data.use_index, day, 2]
                        feed_dict = {self.embedding: embedding, self.factor: feature, self.factor_index: [fac_idx], self.ic: label}
                        new_f, loss_val = sess.run([self.new_f, self.cost], feed_dict=feed_dict)
                        loss_all += loss_val
                        loss_count += 1             
                    avg_loss = loss_all / loss_count
                    if vali_loss_max > avg_loss:
                        vali_loss_max = avg_loss
                        tolerance = 3
                    else:
                        if tolerance == 0:
                            break
                        else:
                            tolerance = tolerance - 1
                    np.random.shuffle(self.rank_day)
                    embedding = self.data.embedding[self.data.use_index, :]
                    for count_day, day in enumerate(self.rank_day):#batch_num
                        feature = self.data.feature_data[self.data.use_index, day, fac_idx]
                        if feature.std() != 0:                            
                            feature = (feature-feature.min()) / (feature.max() - feature.min())
                        else:
                            continue
                        label = self.data.ar_ic[self.data.use_index, day, 2]
                        feed_dict = {self.embedding: embedding, self.factor: feature, self.factor_index: [fac_idx], self.ic: label}
                        _, loss_val = sess.run([self.optimizer, self.cost], feed_dict=feed_dict)
            if not os.path.exists(data_dir+'model_{}'.format(self.param)):
                os.mkdir(data_dir+'model_{}'.format(self.param))
            saver.save(sess, data_dir+'model_{}/logmodel.ckpt'.format(self.param))
            self.test()
            
    
    def test(self):
        saver = tf.train.Saver(max_to_keep=50)
        with tf.Session() as sess:
            saver.restore(sess, data_dir+'{}model_{}/logmodel.ckpt'.format(self.param))
           
            rank_ic = np.zeros((self.day_sample-self.vali_test, 44, 4))
            embedding = self.data.embedding[self.data.use_index, :]
            print('test: model:{}')
            for count_day, day in enumerate(range(self.vali_test, self.day_sample)):#batch_num
                for fac_idx in range(44):
                    feature = self.data.feature_data[self.data.use_index, day, fac_idx]
                    if feature.std() != 0:
                        feature = (feature-feature.min()) / (feature.max() - feature.min())
                    else:
                        continue
                    label = self.data.ar_ic[self.data.use_index, day, 2]
                    feed_dict = {self.embedding: embedding, self.factor: feature, self.factor_index: [fac_idx], self.ic: label}
                    new_f, loss_val = sess.run([self.new_f, self.cost], feed_dict=feed_dict)
                    for ic_idx in range(4):
                        rank_ic[count_day, fac_idx, ic_idx] = stats.spearmanr(self.data.ar_ic[self.data.use_index, day, ic_idx], new_f)[0]
            avg_ic = rank_ic.mean(axis=0)[:, 2]
            print(np.abs(avg_ic).mean())
            pd.DataFrame(np.abs(avg_ic)).to_csv(data_dir+'{}model_{}/test.csv'.format(self.param))              
            
            rank_ic = np.zeros((self.day_sample-self.vali_test, 44, 4))
            for count_day, day in enumerate(range(self.vali_test, self.day_sample)):#batch_num
                for fac_idx in range(44):
                    feature = self.data.feature_data[self.data.use_index, day, fac_idx]
                    for ic_idx in range(4):
                        rank_ic[count_day, fac_idx, ic_idx] = np.nan_to_num(stats.spearmanr(self.data.ar_ic[self.data.use_index, day, ic_idx], feature)[0])
            avg_ic = rank_ic.mean(axis=0)[:, 2]
            print(np.abs(avg_ic).mean())            
            pd.DataFrame(np.abs(avg_ic)).to_csv(data_dir+'{}model_{}/test_baseline.csv'.format(self.param))              
            


if __name__ == '__main__':
    data = Datum()
    data.data_prepare()
    data.get_embedding(sys.argv[1])
    data.supervised_data_prepare(int(sys.argv[2]), int(sys.argv[3]))
    data.ic_prepare(int(sys.argv[2]), int(sys.argv[3]))

    mod = Model(2, sys.argv[1])
    mod.data_initial(data)
    mod.data_split()
    mod.factor_network()    
    mod.training()
    
    
'''
   def factor_network(self):
        learning_rate = self.learning_rate
        
        self.embedding = tf.placeholder(tf.float32, shape=[32, 1], name='embedding')
        self.factor = tf.placeholder(tf.float32, shape=[44, None], name='factor')
        self.factor_index = tf.placeholder(tf.int32, shape=[5], name='factor_index')
        self.label = tf.placeholder(tf.int32, shape=[None, 2], name='lable')
        
        self.u_bias = tf.get_variable('u_bias', shape=[44, 1, 32], initializer=tf.truncated_normal_initializer(stddev=1.0))
        self.weight = tf.get_variable(name='weight', shape=[32, 32], initializer=tf.truncated_normal_initializer(stddev=1.0))
        self.bias = tf.get_variable(name='bias', shape=[32], initializer=tf.zeros_initializer)
        self.project = tf.get_variable(name='project', shape=[32], initializer=tf.truncated_normal_initializer(stddev=1.0))
        
        self.u_bias_select = tf.nn.embedding_lookup(self.u_bias, self.factor_index)
        self.factor_select = tf.nn.embedding_lookup(self.factor, self.factor_index)
        
        concat = []
        
        for i in range(5):
            add1 = tf.transpose(tf.matmul(self.weight, self.embedding))
            add2 = tf.matmul(tf.expand_dims(self.factor_select[i, :], 1), self.u_bias_select[i])
            concat.append(tf.reduce_sum(self.project * tf.tanh(add1 + add2 + self.bias), axis=1, keep_dims=True))
        self.hidden = tf.concat([concat[i] for i in range(5)], axis=1)
        

        all_feature = tf.concat([tf.transpose(self.factor_select), self.hidden], 1)
        self.pred = tf.contrib.layers.fully_connected(
            inputs=all_feature,
            num_outputs=2,  # hidden
            activation_fn=tf.tanh,
            weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
            biases_initializer=tf.zeros_initializer()
        )
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.label))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        
        concat_all = []
        
        for i in range(44):
            add3 = tf.transpose(tf.matmul(self.weight, self.embedding))
            add4 = tf.matmul(tf.expand_dims(self.factor[i, :], 1), self.u_bias[i])
            concat_all.append(tf.reduce_sum(self.project * tf.tanh(add3 + add4 + self.bias), axis=1, keep_dims=True))
        self.evaluation = tf.concat([concat_all[i] for i in range(44)], axis=1)
        
    def training(self):
        epochs = 50
        batch_num = self.train_vali // self.batch_size
        saver = tf.train.Saver(max_to_keep=None)
        with tf.Session() as sess:
            saver.restore(sess, 'model_initial2/logmodel.ckpt')
            for epoch in range(epochs):
                print('epoch: {}'.format(epoch))
                # validation
                new_f_total = []
                for stock_idx in range(self.use_index):
                    price = self.data.price_data[stock_idx, self.train_vali:self.vali_test]
                    feature = self.data.feature_data[stock_idx, self.train_vali:self.vali_test, :]
                    label = self.data.ar_trend[stock_idx, self.train_vali:self.vali_test, :]
                    stock_name = str(self.data.code_tag[stock_idx])
                    embed = np.expand_dims(self.data.embedding[self.data.list_stocks.index(stock_name)], axis=1)
                    factor_index = random.sample(list(range(44)), 5)
                    feed_dict = {self.embedding: embed, self.factor: feature.T, self.factor_index: factor_index, self.label: label}
                    new_f= sess.run(self.evaluation, feed_dict=feed_dict)        
                    new_f_total.append(new_f)
                new_f_total = np.array(new_f_total)
                use_index= self.use_index
                ic = np.zeros((44, self.vali_test-self.train_vali, 4))
                for day in range(0, self.vali_test-self.train_vali):
                    for fac in range(44):
                        for id_idx in range(4):
                            rank_ic = stats.spearmanr(self.data.ar_ic[use_index, day, id_idx], new_f_total[:, day, fac])
                            ic[fac, day, id_idx] = rank_ic[0]
                ic = ic.mean(axis=1)
'''    