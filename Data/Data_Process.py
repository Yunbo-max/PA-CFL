{
    "fileheader.author": "JanKinCai",
}

import pandas as pd
import pandas as pd

# Read the integrated_train_data.csv file
dataset = pd.read_csv('E:\Federated_learning_flower\experiments\Presentation\integrated_train_data_ISMM.csv')

# Get unique market values
markets = dataset['index'].unique()

# Create a dictionary to map regions to numbers
region_mapping = {region: i for i, region in enumerate(markets)}

# Add a new column 'Region Index' to represent the regions with numbers
dataset['Region Index'] = dataset['index'].map(region_mapping)

dataset = dataset.drop(columns=['index'])

# Print the modified dataset
print(dataset)
print(region_mapping)

# Get unique market values
markets = dataset['Region Index'].unique()

print(markets)

#
with h5py.File('market_data.h5', 'w') as f:
    for market in markets:
        market_str = str(market)
        market_data = dataset[dataset['Region Index'] == market]
        print(market_str)
        print(market_data)
        # Store the data as a dataset
        f.create_dataset(market_str, data=market_data.to_numpy())
        # Store the column names as an attribute
        f[market_str].attrs['columns'] = market_data.columns.tolist()
