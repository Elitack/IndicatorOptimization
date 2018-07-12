import os
import time

os.system('python embedding.py 2006010120160101 2016010120170101 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2006010120160101 2016010120170101 year')
'''
os.system('python embedding.py 2005100120151001 2015100120161001 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2005100120151001 2015100120161001 year')
os.system('python embedding.py 2005070120150701 2015070120160701 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2005070120150701 2015070120160701 year')
os.system('python embedding.py 2005040120150401 2015040120160401 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2005040120150401 2015040120160401 year')
'''