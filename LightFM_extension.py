#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:59:12 2023

@author: hoahduong & ridhikaagrawal

Usage:
    $ python3 LightFM_extension.py
"""
import pandas as pd
import pyarrow.parquet as pq
# import numpy as np
# import scipy.sparse as sp 
# from lightfm import lightFM
# from lightfm.data import Dataset
# import time

# importing 
# file_names = ['interactions_train_small']
# file_paths = [f'{x}_indexed_row.parquet' for x in file_names]
# file_paths.append('interactions_test_indexed_row.parquet')
# file_paths = ['people_big.parquet']

file_paths = ['utility_test.parquet', 'utility_train.parquet', 'utility_val.parquet']

utility_test = pd.read_parquet(file_paths[0])
utility_train = pd.read_parquet(file_paths[1])
utility_val = pd.read_parquet(file_paths[2])

print("test size", len(utility_test))
print("train size", len(utility_train))
print("val size", len(utility_val))
