# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-13 13:34:52
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-13 13:36:40


import pandas as pd
import h5py


dataset = pd.read_csv('Clustering_analysis/integrated_train_data_ISMM_test.csv')
markets = dataset['index'].unique()
region_mapping = {region: i for i, region in enumerate(markets)}
dataset['Region Index'] = dataset['index'].map(region_mapping)
dataset = dataset.drop(columns=['index'])

print(dataset)
print(region_mapping)

# Get unique market values
markets = dataset['Region Index'].unique()
with h5py.File('Clustering_analysis/market_data.h5', 'w') as f:
    for market in markets:
        market_str = str(market)
        market_data = dataset[dataset['Region Index'] == market]
        print(market_str)
        print(market_data)
        # Store the data as a dataset
        f.create_dataset(market_str, data=market_data.to_numpy())
        # Store the column names as an attribute
        f[market_str].attrs['columns'] = market_data.columns.tolist()