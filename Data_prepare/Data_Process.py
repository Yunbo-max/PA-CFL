# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 00:22:54
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-11 12:13:12

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import Ridge,LinearRegression,LogisticRegression,ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor,BaggingClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,mean_squared_error,recall_score,confusion_matrix,f1_score,roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


# Hiding the warnings
warnings.filterwarnings('ignore')

dataset =pd.read_csv('./Data/DataCoSupplyChainDataset.csv',encoding='latin-1')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Sort the DataFrame based on the "DateOrders" column
df_sorted = dataset.sort_values(by='order date (DateOrders)')

# If you want to keep the original index order, you can reset the index
df_sorted.reset_index(drop=True, inplace=True)

df_sorted['order date (DateOrders)'] = pd.to_datetime(df_sorted['order date (DateOrders)'])
df_sorted['shipping date (DateOrders)'] = pd.to_datetime(df_sorted['shipping date (DateOrders)'])

# Create a new column 'operation time' representing the difference between order date and shipping date
df_sorted['operation time'] = df_sorted['shipping date (DateOrders)'] - df_sorted['order date (DateOrders)']

# Set the 'DateOrders' column as the index
df_sorted.set_index('order date (DateOrders)', inplace=True)

df_sorted=df_sorted.drop(columns=['shipping date (DateOrders)'])



dataset['Customer Full Name'] = dataset['Customer Fname'] + dataset['Customer Lname']
dataset['TotalPrice'] = dataset['Order Item Quantity'] * dataset[
    'Sales per customer']  # Multiplying item price * Order quantity

data = dataset.drop(
    ['Customer Email', 'Customer Id', 'Customer Password', 'Customer Fname', 'Customer Lname',
      'Product Description', 'Product Image', 'Order Zipcode','Product Status','Order Profit Per Order','Product Price'], axis=1)

data['Customer Zipcode'] = data['Customer Zipcode'].fillna(0)  # Filling NaN columns with zero


data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).day_name()
data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
# data['order_second'] = pd.DatetimeIndex(data['order date (DateOrders)'])

data['shipping_year'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).year
data['shipping_month'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).month
data['shipping_week_day'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).day_name()
data['shipping_hour'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).hour

label_data = data[['shipping_week_day','order_week_day','Customer Full Name','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order State','Order Status','Product Name','shipping date (DateOrders)','Shipping Mode']]


target=data['Sales']

data=data.drop(columns=['shipping_week_day','order_week_day','Customer Full Name','Sales','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order Region','Order State','Order Status','Product Name','shipping date (DateOrders)','Shipping Mode'])


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
print(data.head(30))





# dataset = pd.read_csv('./Data/DataCoSupplyChainDataset.csv', encoding='latin1')

# markets = dataset['index'].unique()
# region_mapping = {region: i for i, region in enumerate(markets)}


# dataset['Region Index'] = dataset['index'].map(region_mapping)

# dataset = dataset.drop(columns=['index'])


# print(dataset)
# print(region_mapping)


# markets = dataset['Region Index'].unique()

# print(markets)


# with h5py.File('../Data/market_data.h5', 'w') as f:
#     for market in markets:
#         market_str = str(market)
#         market_data = dataset[dataset['Region Index'] == market]
#         print(market_str)
#         print(market_data)
#         # Store the data as a dataset
#         f.create_dataset(market_str, data=market_data.to_numpy())
#         # Store the column names as an attribute
#         f[market_str].attrs['columns'] = market_data.columns.tolist()
