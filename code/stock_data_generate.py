import os
import numpy
import pandas as pd

class StockData:
    def __init__(self):
        rootDir = '/data/AMC/06_stock_price_weekly/'
        paths = []
        for file_name in os.listdir(rootDir):
            paths.append(os.path.join(rootDir, file_name))
        paths.sort()
        self.price_file = pd.DataFrame()
        for path in paths:
            csv_price = pd.read_csv(path, encoding='gbk')
            self.price_file = pd.concat([self.price_file, csv_price], ignore_index=True)
            
        rootDir = '/data/AMC/03_stock_features_daily/'
        self.fac_file = []
        dir_name_list = list(os.listdir(rootDir))
        dir_name_list.sort()
        for dir_name in dir_name_list:
            print(dir_name)
            fac_dir = os.path.join(rootDir, dir_name)
            paths = []
            for file_name in os.listdir(fac_dir):
                paths.append(os.path.join(fac_dir, file_name))
            paths.sort()
            fea_file = pd.DataFrame()
            for path in paths:
                csv_price = pd.read_csv(path, encoding='gbk')
                fea_file = pd.concat([fea_file, csv_price])
            self.fac_file.append(fea_file)        
        