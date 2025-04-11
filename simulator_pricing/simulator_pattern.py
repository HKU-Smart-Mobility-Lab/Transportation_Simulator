# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:20:13 2018

@author: kejintao

input information:
1. demand patterns (on minutes)
2. demand databases
3. drivers' working schedule (online/offline time)

** All the inputs are obtained from env, thus we do not need to alter parameters here
"""

from dataclasses import replace
from lib2to3.pgen2 import driver
import numpy as np
import pandas as pd
from copy import deepcopy
import random
from config import *
from path import *
import pickle
import sys
import os

class SimulatorPattern(object):
    def __init__(self):
        # read parameters
        # self.request_file_name = kwargs['request_file_name']
        # self.driver_file_name = kwargs['driver_file_name']
        # orders_grid35_2015-05-04.pickle
        # drivers_grid35_1000.pickle 
        self.request_file_name = os.path.join(data_path, f"orders_grid{env_params['grid_num']}_{env_params['date']}.pickle")
        self.driver_file_name = os.path.join(data_path, f"drivers_grid{env_params['grid_num']}_1000.pickle")

        with open(self.request_file_name, 'rb') as f:
            self.request_all = pickle.load(f)
        with open(self.driver_file_name, 'rb') as f:
            self.driver_info = pickle.load(f)
        self.driver_info = self.driver_info.sample(n=env_params['driver_num'],replace=False, random_state=42)
        print("driver number: ",len(self.driver_info))









