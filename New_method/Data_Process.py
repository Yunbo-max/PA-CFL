# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 00:22:54
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-11 11:57:20


import pandas as pd
import pandas as pd


dataset = pd.read_csv('./Data/DataCoSupplyChainDataset.csv', encoding='latin1')
markets = dataset['Region'].unique()

region_mapping = {region: i for i, region in enumerate(markets)}


dataset['Region Index'] = dataset['index'].map(region_mapping)

dataset = dataset.drop(columns=['index'])


print(dataset)
print(region_mapping)


markets = dataset['Region Index'].unique()

print(markets)


with h5py.File('../Data/market_data.h5', 'w') as f:
    for market in markets:
        market_str = str(market)
        market_data = dataset[dataset['Region Index'] == market]
        print(market_str)
        print(market_data)
        # Store the data as a dataset
        f.create_dataset(market_str, data=market_data.to_numpy())
        # Store the column names as an attribute
        f[market_str].attrs['columns'] = market_data.columns.tolist()
