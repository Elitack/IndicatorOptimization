from datum import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation
import sys
'''
def embedding_cluster(file_param):
    data.get_embedding(file_param)
    af = AffinityPropagation(preference=-50).fit(data.embedding)
    distance = np.zeros((data.embedding.shape[0], data.embedding.shape[0]))
    for i in range(data.embedding.shape[0]):
        for j in range(data.embedding.shape[0]):
            distance[i][j] = np.linalg.norm(data.embedding[i]-data.embedding[j])
    f = open('cluster_result_'+file_param+'.txt', 'w')
    cluster_num = 0
    for idx in af.cluster_centers_indices_:
        f.write('new cluster\n')
        cluster_idx = np.where(af.labels_ == af.labels_[idx])[0]
        sli = distance[idx]
        arg_rank = np.argsort(sli[cluster_idx])
        for arg in arg_rank:
            arg_true = cluster_idx[arg]
            if arg_true >= len(data.use_index):
                continue
            code = data.list_stocks[arg_true]       
            try:
                name = data.dict_code2name[code].split(';')[-2]
            except:
                name = 'Not know'
            f.write(str(cluster_num)+','+str(arg_true)+', '+code+', '+name+'\n')
        cluster_num = cluster_num + 1
        f.write('\n\n\n\n')
'''
        
def get_data(start_date, end_date, stock_code):
    price_csv = pd.read_csv('/data/zhige_data/embedding_simpyear/stock_data/'+str(stock_code)+'_price.csv')
    dates = np.array(price_csv['TradingDay'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_price = np.array(price_csv['ClosePrice'][select_index])
    
    feature_csv = pd.read_csv('/data/zhige_data/embedding_simpyear/stock_data/'+str(stock_code)+'_feature.csv')
    dates = np.array(feature_csv['trading_day'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_feature = np.array(feature_csv)[select_index, 1:]  
    
    return return_price, return_feature
    
# stock select: 2016,2015...
# season_select: all, year
# emb:2012, 2013...

def embedding(emb, stock_select, season_select):
    day_feature = get_data(stock_select[0], stock_select[1], 1)[1].shape[0]
    day_price = get_data(stock_select[0], stock_select[1], 1)[0].shape[0]
    select_num_set = set()
    for num in num_set:
        price, feature = get_data(stock_select[0], stock_select[1], int(num))
        if price.shape[0] == day_price and feature.shape[0] == day_feature:
            s_price, s_feature = get_data(stock_select[1], stock_select[1]+20, int(num))
            if s_price.shape[0] >= 11 and s_feature.shape[0] >= 11:
                select_num_set.add(num)   

    stable_stock_code = set()
    for num in select_num_set:
        code = num
        for _ in range(6-len(code)):
            code = '0' + code     
        if code in stable_stock_code:
            continue
        else:
            stable_stock_code.add(code)    
    
    stable_index = []
    for idx, stock in enumerate(data.list_stocks):
        if stock in stable_stock_code:
            stable_index.append(idx)
    stable_index = np.array(stable_index)    
    list_stable_stocks = [data.list_stocks[i] for i in stable_index]   
    
    weight_matrix = np.copy(data.weight_matrix)
    start_season = np.where(np.array(data.select_date) < emb[0])[0][-1]
    end_season = np.where(np.array(data.select_date) < emb[1])[0][-1]
    if season_select == 'all':
        weight_matrix_total = np.sum(weight_matrix[start_season:end_season], axis=0)
        print(weight_matrix[start_season:end_season].shape)
    else:
        index = np.array(range(start_season+(start_season+1)%2, end_season+end_season%2, 2))
        weight_matrix_total = np.sum(weight_matrix[index], axis=0)
        print(weight_matrix[index].shape)        

    weight_matrix_total = weight_matrix_total[stable_index, :]

    for weight_index, weight in enumerate([weight_matrix_total]):
        arr_tmp = []
        for stock_index in range(len(list_stable_stocks)):
            a_edge_index = np.where(weight[stock_index] != 0)[0]
            for edge_index in a_edge_index:
                arr_tmp.append([stock_index, len(list_stable_stocks) + edge_index, weight[stock_index, edge_index]])
        arr_tmp = np.array(arr_tmp)
        pd_tmp = pd.DataFrame(arr_tmp)
        pd_tmp[0] = pd_tmp[0].astype(int)
        pd_tmp[1] = pd_tmp[1].astype(int)
        path = '/data/zhige_data/embedding_simpyear/graph/graph_{}_{}_{}.csv'.format(
            str(emb[0])+str(emb[1]), str(stock_select[0])+str(stock_select[1]), season_select)
        pd_tmp.to_csv(path, index=False, sep=' ')


        nx_G = nx.read_edgelist(path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        nx_G = nx_G.to_undirected()
        G = graph.Graph(nx_G, False, 1, 1)
        G.preprocess_transition_probs()

        walks = G.simulate_walks(100, 200)
        walks = [list(map(str, walk)) for walk in walks]
        '''
        true_walks = []
        for walk in walks:
            true_walk = []
            for ele in walk:
                if int(ele) < len(list_stable_stocks):
                    true_walk.append(ele)
            true_walks.append(true_walk)
        '''
        model = Word2Vec(walks, size=32, window=6, min_count=0, sg=1, workers=2, iter=30)
        model.wv.save_word2vec_format('/data/zhige_data/embedding_simpyear/embedding/embedding_{}_{}_{}.emb'.format(
            str(emb[0])+str(emb[1]), str(stock_select[0])+str(stock_select[1]), season_select))
        
    np.save('/data/zhige_data/embedding_simpyear/embedding/stable_index_{}_{}_{}.npy'.format(
        str(emb[0])+str(emb[1]), str(stock_select[0])+str(stock_select[1]), season_select), stable_index)
        
    # embedding_cluster('{}_{}_{}'.format(
        # str(emb[0])+str(emb[1]), str(stock_select[0])+str(stock_select[1]), season_select))
    
if __name__ == '__main__':
    num_set = set()
    for filename in os.listdir('/data/zhige_data/embedding_simpyear/stock_data/'):
        num = filename.split('_')[0]
        if num in num_set:
            continue
        else:
            num_set.add(num)    
    data = Datum('{}_{}_{}'.format(sys.argv[1], sys.argv[2], sys.argv[3]))
    data.data_prepare() 
    embedding([int(sys.argv[1][:8]), int(sys.argv[1][8:])], [int(sys.argv[2][:8]), int(sys.argv[2][8:])], sys.argv[3])

