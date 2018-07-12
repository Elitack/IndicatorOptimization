import os
import time

os.system('python embedding.py 2005010120150101 2015010120160101 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2005010120150101 2015010120160101 year')
'''
os.system('python embedding.py 2004100120141001 2014100120151001 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2004100120141001 2014100120151001 year')
os.system('python embedding.py 2004070120140701 2014070120150701 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2004070120140701 2014070120150701 year')
os.system('python embedding.py 2004040120140401 2014040120150401 year')
os.system('CUDA_VISIBLE_DEVICES=0 python model.py 2004040120140401 2014040120150401 year')
'''