import os

# os.system('python model-noemb.py 2014emb 20140101 20150101')
# os.system('python model-noemb.py 2015emb 20150101 20160101')
# os.system('python model-noemb.py 2016emb 20160101 20170101')

os.system('python model-1to1.py 2014emb 20140101 20150101')
os.system('python model-1to1.py 2015emb 20150101 20160101')
os.system('python model-1to1.py 2016emb 20160101 20170101')