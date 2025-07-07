# -*- encoding: UTF-8 -*-
import yaml
import os
import akshare as ak

# 全局变量
config = None
top_list = None

def init():
    global config
    global top_list
    root_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    config_file = os.path.join(root_dir, 'config.yaml')
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    # df = ak.stock_lhb_stock_statistic_em(symbol="近三月")
    # mask = (df['买方机构次数'] > 1)  # 机构买入次数大于1
    # df = df.loc[mask]
    # top_list = df['代码'].tolist()


def get_config():
    global config
    if config is None:
        init()
    return config

# 自动初始化
init()