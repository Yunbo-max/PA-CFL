# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-13 13:29:52
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-13 13:32:58

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.model_selection import train_test_split

# Hiding the warnings
warnings.filterwarnings('ignore')

dataset =pd.read_csv('/Users/yunbo/Documents/GitHub/PFL_Optimiozation/Data/DataCoSupplyChainDataset.csv',encoding='iso-8859-1')
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# print(dataset.head(30))

dataset['Customer Full Name'] = dataset['Customer Fname'].astype(str) + dataset['Customer Lname'].astype(str)
dataset['TotalPrice'] = dataset['Order Item Quantity'] * dataset['Sales per customer']  # Multiplying item price * Order quantity
data = dataset.drop(['Customer Email', 'Customer Id', 'Customer Password', 'Customer Fname', 'Customer Lname',
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


label_data = data[['Order Region','shipping_week_day','order_week_day','Customer Full Name','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order State','Order Status','Product Name','shipping date (DateOrders)','Shipping Mode']]

data['index'] = data['Order Region']
target=data['Sales']

data=data.drop(columns=['shipping_week_day','order_week_day','Customer Full Name','Sales','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order State','Order Status','Product Name','shipping date (DateOrders)','Shipping Mode','Order Region','Sales per customer'])


def Labelencoder_feature(x):
    le=LabelEncoder()
    x=le.fit_transform(x)
    return x

# Apply label encoding to remaining columns
data_encoded = label_data.apply(Labelencoder_feature)
data = pd.concat([data_encoded, data], axis=1)
data_final=data[['Order Id', 'Order Customer Id', 'Order Item Id',
 'Order Item Product Price' ,'Department Id', 'Order Item Quantity',
 'Category Id' ,'shipping_month' ,'Benefit per order' ,'Order Item Total',
 'Product Card Id', 'Product Name', 'Order Item Cardprod Id' ,
 'Order State', 'Product Category Id', 'order_week_day', 'shipping_year',
 'Category Name', 'order_month', 'order_year' ,'Order Item Discount',
 'Department Name', 'Market', 'Order City',
 'Days for shipment (scheduled)' ,'Customer Segment', 'Customer Full Name','index']]

train_data = data_final
xs=train_data.loc[:, train_data.columns != 'Sales']
ys=target
xs_train, xs_test,ys_train,ys_test = train_test_split(xs,ys,test_size = 0.3, random_state = 42)
train_data['Sales'] = ys.values

# Save the integrated DataFrame to CSV
train_data.to_csv('integrated_train_data_ISMM_test.csv', index=False)