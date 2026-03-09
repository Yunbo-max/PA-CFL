# -*- coding: utf-8 -*-
# Unified data preprocessing pipeline for PA-CFL.
# Produces consistent train/test splits across all methods (baselines + PA-CFL).

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

REGION_MAP = {
    0: "Southeast Asia",
    1: "South Asia",
    2: "Oceania",
    3: "Eastern Asia",
    4: "West Asia",
    5: "West of USA",
    6: "US Center",
    7: "West Africa",
    8: "Central Africa",
    9: "North Africa",
    10: "Western Europe",
    11: "Northern Europe",
    12: "Central America",
    13: "Caribbean",
    14: "South America",
    15: "East Africa",
    16: "Southern Europe",
    17: "East of USA",
    18: "Canada",
    19: "Southern Africa",
    20: "Central Asia",
    21: "Eastern Europe",
    22: "South of  USA",
}

# The 14 regions used in experiments
ACTIVE_REGIONS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 14, 16, 17, 22]


def _label_encode(x):
    le = LabelEncoder()
    return le.fit_transform(x)


def preprocess_data(data):
    """Feature engineering and selection. Returns (features_df, labels_series)."""
    data = data.copy()

    data['Customer Full Name'] = data['Customer Fname'].astype(str) + data['Customer Lname'].astype(str)
    data['TotalPrice'] = data['Order Item Quantity'] * data['Sales per customer']

    data = data.drop(
        ['Customer Email', 'Customer Id', 'Customer Password', 'Customer Fname',
         'Customer Lname', 'Product Description', 'Product Image', 'Order Zipcode',
         'Product Status', 'Order Profit Per Order', 'Product Price'], axis=1)

    data['Customer Zipcode'] = data['Customer Zipcode'].fillna(0)

    # Temporal features
    data['order_year'] = pd.DatetimeIndex(data['order date (DateOrders)']).year
    data['order_month'] = pd.DatetimeIndex(data['order date (DateOrders)']).month
    data['order_week_day'] = pd.DatetimeIndex(data['order date (DateOrders)']).day_name()
    data['order_hour'] = pd.DatetimeIndex(data['order date (DateOrders)']).hour
    data['shipping_year'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).year
    data['shipping_month'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).month
    data['shipping_week_day'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).day_name()
    data['shipping_hour'] = pd.DatetimeIndex(data['shipping date (DateOrders)']).hour

    label_data = data[['shipping_week_day', 'order_week_day', 'Customer Full Name',
                        'Type', 'Delivery Status', 'Category Name', 'Customer City',
                        'Customer Country', 'Customer Segment', 'Customer State',
                        'Customer Street', 'Department Name', 'Market', 'Order City',
                        'Order Country', 'order date (DateOrders)', 'Order State',
                        'Order Status', 'Product Name', 'Shipping Mode']]

    label_y = data['Sales']

    data = data.drop(columns=[
        'shipping date (DateOrders)', 'Sales per customer', 'Order Region',
        'shipping_week_day', 'order_week_day', 'Customer Full Name', 'Sales',
        'Type', 'Delivery Status', 'Category Name', 'Customer City',
        'Customer Country', 'Customer Segment', 'Customer State',
        'Customer Street', 'Department Name', 'Market', 'Order City',
        'Order Country', 'order date (DateOrders)', 'Order State',
        'Order Status', 'Product Name', 'Shipping Mode',
        'Order Item Product Price', 'TotalPrice'],
        errors='ignore')

    # Remove duplicate column references
    data = data.loc[:, ~data.columns.duplicated()]

    data_encoded = label_data.apply(_label_encode)
    data = pd.concat([data_encoded, data], axis=1)

    SELECTED_FEATURES = [
        'Benefit per order', 'Order Id', 'Order Customer Id', 'Order Item Id',
        'Order Item Quantity', 'Department Id', 'Order Item Total', 'Category Id',
        'shipping_month', 'Product Card Id', 'Product Name',
        'Order Item Cardprod Id', 'order date (DateOrders)', 'Order State',
        'Order Item Discount', 'Market', 'Department Name', 'order_week_day',
        'Product Category Id', 'order_year', 'order_month', 'Category Name',
        'shipping_year', 'Order City', 'Days for shipment (scheduled)',
        'Customer Segment', 'Customer Full Name'
    ]

    data_final = data[[c for c in SELECTED_FEATURES if c in data.columns]]
    return data_final, label_y


def load_and_prepare_data(data_path, output_path=None, test_size=0.2, random_state=42):
    """Load raw CSV, preprocess per region, return dict of {region: {train, test}} tensors.

    Args:
        data_path: Path to DataCoSupplyChainDataset.csv
        output_path: If provided, save processed datasets as pickle
        test_size: Fraction for test split
        random_state: Random seed for reproducibility

    Returns:
        datasets: dict mapping region_name -> {
            'train_features': Tensor, 'train_labels': Tensor,
            'test_features': Tensor, 'test_labels': Tensor,
            'scaler': fitted MinMaxScaler
        }
    """
    dataset = pd.read_csv(data_path, encoding='latin-1')
    datasets = {}

    for region, data_region in dataset.groupby('Order Region'):
        X_train_raw, X_test_raw = train_test_split(
            data_region, test_size=test_size, random_state=random_state)

        X_train, y_train = preprocess_data(X_train_raw)
        X_test, y_test = preprocess_data(X_test_raw)

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)
        X_test_scaled = scaler.transform(X_test.values)

        datasets[region] = {
            'train_features': torch.tensor(X_train_scaled, dtype=torch.float32),
            'train_labels': torch.tensor(y_train.values, dtype=torch.float32),
            'test_features': torch.tensor(X_test_scaled, dtype=torch.float32),
            'test_labels': torch.tensor(y_test.values, dtype=torch.float32),
            'scaler': scaler,
            'num_features': X_train_scaled.shape[1],
        }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(datasets, f)
        print(f"Saved processed datasets to {output_path}")

    return datasets
