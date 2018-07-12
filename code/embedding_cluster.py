from datum import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation
import sys
import io

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

        
        
if __name__ == '__main__':
    num_set = set()
    for filename in os.listdir('/data/zhige_data/huaxia_embedding/stock_data/'):
        num = filename.split('_')[0]
        if num in num_set:
            continue
        else:
            num_set.add(num)    
    data = Datum('{}_{}_{}'.format(sys.argv[1], sys.argv[2], sys.argv[3]))
    data.data_prepare() 

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