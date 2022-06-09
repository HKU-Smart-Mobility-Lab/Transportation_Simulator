# # from utilities import *
# import sys
#
# import numpy as np
# import pandas as pd
#
# import pickle
#
# records = pickle.load(open('dataset.pickle','rb'))
#
# for time in records.keys():
#     for order in records[time]:
#             print(order)
#             sys.exit()
#
#
# pickle.dump(records,open('dataset.pickle','wb'))

# from azureml.opendatasets import NycTlcYellow
#
# from datetime import datetime
# from dateutil import parser
#
#
# end_date = parser.parse('2015-05-31')
# start_date = parser.parse('2015-05-01')
# nyc_tlc = NycTlcYellow(start_date=start_date, end_date=end_date)
#
# nyc_tlc_df = nyc_tlc.to_pandas_dataframe()
#
# nyc_tlc_df.info()
# print(nyc_tlc_df)
import sys

import pickle
data  = pickle.load(open("input/NYU_May.pickle",'rb'))

for date in data.keys():
    print(data[date])
    # for time in data[date].keys():
    #     print(date,time,data[date][time])
    #     sys.exit()
