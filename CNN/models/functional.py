"""Functional interface"""

import warnings
import math

import torch

from .ori_dropout import *
# from .ori_sparse_dropout  import *
# from .spatial_dropout  import *
# from .ori_sparse_dropout  import *
# #from .half_dropout import *
#from .crossmap_dropout import *

# Activation functions
def dropout(input, p=10, training=False, inplace=False):  # ori,fdd drop
    return Dropout.apply(input, p, training, inplace)

# def dropout(input, p=10, epoch=1, total_epoch=1, training=False, inplace=False):  #fdd drop
#     return Dropout.apply(input, p, epoch, total_epoch, training, inplace)

# def dropout(input, epoch, p=10, training=False, inplace=False):  #ori,cnn,low and high drop
#     return Dropout.apply(input, epoch, p, training, inplace)

# def dropout(input, p=10, linear_weight=0, training=False, inplace=False): #weight drop
#     return Dropout.apply(input, p, linear_weight, training, inplace)
