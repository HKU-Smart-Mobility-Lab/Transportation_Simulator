# from utilities import *
import sys

import numpy as np
import pandas as pd

import pickle

records = pickle.load(open('dataset.pickle','rb'))

for time in records.keys():
    for order in records[time]:
            print(order)
            sys.exit()


pickle.dump(records,open('dataset.pickle','wb'))