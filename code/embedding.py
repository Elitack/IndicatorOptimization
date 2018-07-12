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

data_dir = '../data/'


def get_data(start_date, end_date, stock_code):
    price_csv = pd.read_csv(data_dir + 'stock_data/'+str(stock_code)+'_price.csv')
    dates = np.array(price_csv['TradingDay'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_price = np.array(price_csv['ClosePrice'][select_index])
    
    feature_csv = pd.read_csv(data_dir + 'stock_data/'+str(stock_code)+'_feature.csv')
    dates = np.array(feature_csv['trading_day'])
    for count in range(len(dates)):
        dates[count] = int(''.join(str(dates[count]).split()[0].split('-')))
    select_index = np.where((dates >= start_date) & (dates < end_date))[0]
    return_feature = np.array(feature_csv)[select_index, 1:]  
    
    return return_price, return_feature
    
# stock select: 2016,2015...
# season_select: all, year
# emb:2012, 2013...

def embedding(emb, output_name):
    weight_matrix = np.copy(data.weight_matrix)
    start_season = np.where(np.array(data.select_date) < emb[0])[0][-1]
    end_season = np.where(np.array(data.select_date) < emb[1])[0][-1]

    index = np.array(range(start_season+(start_season+1)%2, end_season+end_season%2, 2))
    weight_matrix_total = np.sum(weight_matrix[index], axis=0)
    print(weight_matrix[index].shape)        

    arr_tmp = []
    for stock_index in range(len(data.list_stocks)):
        a_edge_index = np.where(weight_matrix_total[stock_index] != 0)[0]
        for edge_index in a_edge_index:
            arr_tmp.append([stock_index, len(data.list_stocks) + edge_index, weight_matrix_total[stock_index, edge_index]])
    arr_tmp = np.array(arr_tmp)
    pd_tmp = pd.DataFrame(arr_tmp)
    pd_tmp[0] = pd_tmp[0].astype(int)
    pd_tmp[1] = pd_tmp[1].astype(int)
    path = data_dir + 'graph/{}.csv'.format(output_name)
    pd_tmp.to_csv(path, index=False, sep=' ')


    nx_G = nx.read_edgelist(path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    nx_G = nx_G.to_undirected()
    G = graph.Graph(nx_G, False, 1, 1)
    G.preprocess_transition_probs()

    walks = G.simulate_walks(200, 200)
    walks = [list(map(str, walk)) for walk in walks]
    
    model = Word2Vec(true_walks, size=32, window=6, min_count=0, sg=1, workers=2, iter=30)
    model.wv.save_word2vec_format(data_dir + 'embedding/{}.emb'.format(output_name))
        
    # embedding_cluster('{}_{}_{}'.format(
        # str(emb[0])+str(emb[1]), str(stock_select[0])+str(stock_indexselect[1]), season_select))
    
if __name__ == '__main__':
    data = Datum()
    data.data_prepare() 
    embedding([int(sys.argv[1][:8]), int(sys.argv[1][8:])] , sys.argv[2])

