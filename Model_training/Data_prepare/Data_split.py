# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-02-11 00:22:54
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-02-12 11:31:46
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import os
import pickle

import torch
from torch.utils.data import Dataset
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

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


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def preprocess_data(data):

 
    # Add new features
    data['TotalPrice'] = data['Order Item Quantity'] * data['Sales per customer']
    data['Customer Zipcode'] = data['Customer Zipcode'].fillna(0)
    data['operation time'] = (pd.DatetimeIndex(data['shipping date (DateOrders)']) - pd.DatetimeIndex(data['order date (DateOrders)'])).total_seconds() / (60*60)  

    data = data.drop(
    ['Customer Email', 'Customer Id', 'Customer Password', 'Customer Fname', 'Customer Lname',
      'Product Description', 'Product Image', 'Order Zipcode','Product Status','Order Profit Per Order','Product Price','shipping date (DateOrders)'], axis=1)


    # Extract datetime features
    data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
    data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
    data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).day_name()
    data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
    
    # Label encoding for categorical columns
    label_encoder = LabelEncoder()
    categorical_columns = ['Delivery Status', 'Order Status', 'Shipping Mode','Type','Category Name','Customer City','Customer Country', 'Customer Segment','Customer State','Customer Street','Department Name', 'Market','Order City','Order Country','Order State','Product Name','order_week_day',]
    
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    
    # # One-hot encoding for category_data2
    # category_data2 = pd.get_dummies(data[['Customer Full Name','Type','Category Name','Customer City','Customer Country',
    #                                        'Customer Segment','Customer State','Customer Street','Department Name',
    #                                        'Market','Order City','Order Country','order_week_day',
    #                                        'Order State','Product Name']])
    
    # Standardize numerical features
    scaler = StandardScaler()
    data[data.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))

    
    # Extract label_data
    label_data = data['Sales']
    
    # Drop irrelevant columns
    data = data.drop(columns=['Sales','order date (DateOrders)','Order Region'])

    # Add new features and preprocess data
    
    print("After preprocessing:")
    print(data.shape)

    return data, label_data


def split_and_save_dataset(dataset, folder_path):
    datasets = {}
    for region, data_region in dataset.groupby('Order Region'):
        X_train=0
        X_test = 0

        # Split data
        X_train, X_test = train_test_split(data_region, test_size=0.2, random_state=42)
        
        # Preprocess train data
        X_train, train_label_data = preprocess_data(X_train)
        
        # Preprocess test data
        X_test, test_label_data = preprocess_data(X_test)
        
        # Concatenate features and labels into a single DataFrame
        train_data = pd.concat([X_train, train_label_data], axis=1)
        test_data = pd.concat([X_test, test_label_data], axis=1)
        
        # Print the shape of the data for each region
        print(f"Region: {region}, Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        
        # Create CustomDataset objects for train and test data
        train_dataset = CustomDataset(X_train.values, train_label_data.values)
        test_dataset = CustomDataset(X_test.values, test_label_data.values)
        
        # Store datasets for each region separately
        datasets[region] = {'train': train_dataset, 'test': test_dataset}
    
    # Save datasets to disk
    file_path = os.path.join(folder_path, 'datasets.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(datasets, f)




# Example usage:
if __name__ == "__main__":
    # Read the dataset
    dataset = pd.read_csv('./Data/DataCoSupplyChainDataset.csv', encoding='latin-1')
    
    # Specify the directory where you want to save the file
    folder_path = 'Model_training/Data_prepare'
    
    # Ensure that the directory exists, create it if necessary
    os.makedirs(folder_path, exist_ok=True)
    
    # Split and save datasets
    split_and_save_dataset(dataset, folder_path)
