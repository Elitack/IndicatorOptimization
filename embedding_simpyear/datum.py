from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import graph
import networkx as nx

data_dir = '../data/'

def get_data(start_date, end_date, stock_code):
    # get the data, the stock_data directory is extracted from /data/AMC/stock_feature and /data/AMC/stock_price
    price_csv = pd.read_csv(data_dir + 'stock_data/'+str(int(stock_code))+'_price.csv')
    dates = np.array(price_csv['TradingDay'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_price = np.array(price_csv['ClosePrice'][select_index])
    
    feature_csv = pd.read_csv(data_dir + 'stock_data/'+str(int(stock_code))+'_feature.csv')
    dates = np.array(feature_csv['trading_day'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_feature = np.array(feature_csv)[select_index, 1:]  
    return return_price, return_feature

class Datum:
    def __init__(self, param=None):
        self.list_funds = []
        self.list_stocks = []
        self.embedding = np.array([0])
        self.dict_code2name = {}
        self.price_data = np.array([0])
        self.feature_data = np.array([0])
        self.code_tag = []
        self.dimension = 32
        # indicator name
        self.indicator = list(pd.read_csv(data_dir + 'stock_data/1_feature.csv').columns)[1:]
        
        if param is not None:
            self.param = param
            # stat_date, end_date: date for indicator optimization model
            self.start_date = int(self.param.split('_')[1][:8])
            self.end_date = int(self.param.split('_')[1][8:])
            
            self.select_date = []
            # emb_start, emb_end: date for embedding
            self.emb_start = int(self.param.split('_')[0][:8]) // 10000
            self.emb_end = int(self.param.split('_')[0][8:]) // 10000
            for year in range(self.emb_start * 10000, self.emb_end * 10000, 10000):
                for month_day in [0, 400, 700, 1000]:
                    self.select_date.append(year + month_day)
            self.select_date.append(self.emb_end * 10000)     
            
    def data_prepare(self):
        # holding data to matrix
        fundhold = pd.read_csv(data_dir + 'mutualfundholding.csv')

        fund = np.array(fundhold)[:, 0]
        date = np.array(fundhold)[:, 1]
        stock = np.array(fundhold)[:, 3]
        value = np.array(fundhold)[:, 4]

        index = np.where((date >= self.emb_start*10000) & (date <= self.emb_end*10000))[0]

        raw_funds = fund[index]
        raw_dates = date[index]
        raw_stocks = stock[index]
        raw_values = value[index]

        list_funds = []
        list_dates = []
        list_stocks = []
        for fund in raw_funds:
            if fund not in list_funds:
                list_funds.append(fund)
        for date in raw_dates:
            if date not in list_dates:
                list_dates.append(date)
        for stock in raw_stocks:
            stock = stock.split('.')[0]
            if stock not in list_stocks:
                list_stocks.append(stock)

        self.list_funds = list_funds
        self.list_stocks = list_stocks

        select_date = np.array(self.select_date)[:-1]

        self.list_funds = list_funds
        self.list_stocks = list_stocks      
        self.weight_matrix = np.zeros((len(select_date), len(self.list_stocks), len(self.list_funds)))

        for ind in range(len(raw_funds)):
            fund_index = self.list_funds.index(raw_funds[ind])
            stock_index = self.list_stocks.index(raw_stocks[ind].split('.')[0])
            time_index = np.where(select_date < raw_dates[ind])[0][-1]
            self.weight_matrix[time_index, stock_index, fund_index] = raw_values[ind]                
            
        # stock code to Chinese
        industry = pd.read_csv(data_dir + 'industry.csv')
        for ele in np.array(industry):
            ele[0] = str(ele[0])
            for _ in range(6-len(ele[0])):
                ele[0] = '0' + ele[0]        
            if not ele[0] in self.dict_code2name:
                self.dict_code2name[ele[0]] = ele[4]+'-'+ele[2]+';'
            else:
                name = ele[4] + '-' + ele[2]
                if name not in self.dict_code2name[ele[0]].split(';'):
                    self.dict_code2name[ele[0]] += name+';'
        
    def graph2file(self):
        # matrix to csv
        for weight_index, weight in enumerate(self.weight_matrix):
            arr_tmp = []
            for stock_index in range(len(self.list_stocks)):
                a_edge_index = np.where(weight[stock_index] != 0)[0]
                for edge_index in a_edge_index:
                    arr_tmp.append([stock_index, len(self.list_stocks) + edge_index, weight[stock_index, edge_index]])
            arr_tmp = np.array(arr_tmp)
            pd_tmp = pd.DataFrame(arr_tmp)
            pd_tmp[0] = pd_tmp[0].astype(int)
            pd_tmp[1] = pd_tmp[1].astype(int)
            pd_tmp.to_csv(data_dir + 'graph/graph_' + str(weight_index) + '.csv', index=False, sep=' ')
            
    def get_embedding(self):
        # get the created embedding
        total_embedding = np.array(pd.read_csv(data_dir + 'embedding/embedding_'+self.param+'.emb', header=None, sep=' ', skiprows=1))
        use_index = np.load(data_dir + 'embedding/stable_index_'+self.param+'.npy')
        self.list_stocks = [self.list_stocks[i] for i in use_index]
        self.embedding = np.zeros((len(self.list_funds)+len(self.list_stocks), total_embedding.shape[1]-1))
        for emb in total_embedding:
            self.embedding[int(emb[0])] = emb[1:]
            
    def supervised_data_prepare(self):
        # price and feature data save
        self.price_data = []
        self.feature_data = []
        for count, code in enumerate(self.list_stocks):
            a_p, a_f = get_data(self.start_date, self.end_date, int(code))
            self.price_data.append(a_p)
            self.feature_data.append(a_f)
        self.price_data = np.array(self.price_data)
        self.feature_data = np.array(self.feature_data).astype(float)
        
    def ic_prepare(self):
        # calculated correlation(IC)
        day_need = get_data(self.start_date, self.end_date, 1)[0].shape[0]
        self.ar_ic = np.zeros((len(self.list_stocks), day_need, 4))
        # mkt1, 3, 5, 10
        for count, code in enumerate(self.list_stocks):
            a_p, a_f = get_data(self.start_date, self.end_date+20, code)
            a_p = a_p[:day_need+11]
            for day in range(day_need):
                for look_count, look_after in enumerate([1, 3, 5, 10]):
                    self.ar_ic[count, day, look_count] = (a_p[day+1+look_after] - a_p[day+1]) / a_p[day+1]
                    

                
        
if __name__ == "__main__":
    data = Datum()
