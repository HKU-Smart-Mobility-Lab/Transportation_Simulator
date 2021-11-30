import pandas as pd
import numpy as np
from path import *
import pickle

records = pickle.load(open(data_path + 'toy_records' + '.pickle', 'rb'))
print(records)


