import pickle

# 指定 .pickle 文件的路径
file_path = 'matching_strategy_baseperformance_record_test_instant_reward_no_subway.pickle'
txt_file_path = 'matching_strategy_baseperformance_record_test_instant_reward_no_subway.txt'

# 打开并读取 .pickle 文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 将数据写入 .txt 文件
with open(txt_file_path, 'w') as txt_file:
    for key, value in data.items():
        txt_file.write(f"{key}: {value}\n")