# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-12 14:58:12
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 15:00:22
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import pickle
import h5py

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):

    data['Customer Full Name'] = data['Customer Fname'].astype(str) + data['Customer Lname'].astype(str)
    data['TotalPrice'] = data['Order Item Quantity'] * data[
        'Sales per customer']  # Multiplying item price * Order quantity



    data = data.drop(
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


    label_data = data[['shipping_week_day','order_week_day','Customer Full Name','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order State','Order Status','Product Name','Shipping Mode']]


    label_y=data['Sales']

    data=data.drop(columns=['shipping date (DateOrders)','Sales per customer','Order Region','shipping_week_day','order_week_day','Customer Full Name','Sales','Type','Delivery Status','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Name','Market','Order City','Order Country','order date (DateOrders)','Order Region','Order State','Order Status','Product Name','shipping date (DateOrders)','Shipping Mode','Order Region','Order Item Product Price','TotalPrice','order date (DateOrders)'])



    def Labelencoder_feature(x):
        le=LabelEncoder()
        x=le.fit_transform(x)
        return x



    # Apply label encoding to remaining columns
    data_encoded = label_data.apply(Labelencoder_feature)

    data = pd.concat([data_encoded, data], axis=1)

    data_final=data[['Benefit per order', 'Order Id', 'Order Customer Id', 'Order Item Id',
    'Order Item Quantity', 'Department Id', 'Order Item Total', 'Category Id',
    'shipping_month', 'Product Card Id', 'Product Name',
    'Order Item Cardprod Id', 'order date (DateOrders)', 'Order State',
    'Order Item Discount' ,'Market', 'Department Name', 'order_week_day',
    'Product Category Id', 'order_year' ,'order_month' ,'Category Name',
    'shipping_year' ,'Order City' ,'Days for shipment (scheduled)',
    'Customer Segment' ,'Customer Full Name']]
    # Add new features and preprocess data
    
    print("After preprocessing:")
    print(data_final.head(10))

    return data_final, label_y

def split_and_save_dataset(dataset, folder_path):
    datasets = {}
    for region, data_region in dataset.groupby('Order Region'):
        X_train, X_test = train_test_split(data_region, test_size=0.2, random_state=42)
        X_train, train_label_data = preprocess_data(X_train)
        X_test, test_label_data = preprocess_data(X_test)
        
        train_data = pd.concat([X_train, train_label_data], axis=1)
        test_data = pd.concat([X_test, test_label_data], axis=1)
        
        train_dataset = CustomDataset(X_train.values, train_label_data.values)
        test_dataset = CustomDataset(X_test.values, test_label_data.values)
        
        datasets[region] = {'train': train_dataset, 'test': test_dataset}
    
    file_path = os.path.join(folder_path, 'datasets.h5')
    save_datasets_h5(datasets, file_path)

def save_datasets_h5(datasets, file_path):
    with h5py.File(file_path, 'w') as hf:
        for region, data in datasets.items():
            region_group = hf.create_group(region)
            for key, dataset in data.items():
                dataset_group = region_group.create_group(key)
                dataset_group.create_dataset('features', data=dataset.features)
                dataset_group.create_dataset('labels', data=dataset.labels)

if __name__ == "__main__":
    dataset = pd.read_csv('./Data/DataCoSupplyChainDataset.csv', encoding='latin-1')
    folder_path = 'Model_training/Data_prepare'
    os.makedirs(folder_path, exist_ok=True)
    split_and_save_dataset(dataset, folder_path)
