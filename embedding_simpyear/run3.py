import os
import time

os.system('python embedding.py 2004010120140101 2014010120150101 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2004010120140101 2014010120150101 year')
'''
os.system('python embedding.py 2003100120131001 2013100120141001 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2003100120131001 2013100120141001 year')
os.system('python embedding.py 2003070120130701 2013070120140701 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2003070120130701 2013070120140701 year')
os.system('python embedding.py 2003040120130401 2013040120140401 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2003040120130401 2013040120140401 year')
'''