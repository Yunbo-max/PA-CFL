import pandas as pd
import re
import os


# Read the log file
with open(
    "../PFL_Optimiozation/Clusters_Computation_cuda/optimised_fl_mac.log", "r"
) as file:
    log_data = file.readlines()

# Extract data from the log
variances_data = []
means_data = []
std_dev_data = []

# Initialize variables
variances_data_started = False
means_data_started = False
std_dev_data_started = False

for line in log_data:
    if "Variances Dataframe" in line:
        variances_data_started = True
        means_data_started = False
        std_dev_data_started = False
        continue
    elif "Means Dataframe" in line:
        variances_data_started = False
        means_data_started = True
        std_dev_data_started = False
        continue
    elif "Standard Deviations Dataframe" in line:
        variances_data_started = False
        means_data_started = False
        std_dev_data_started = True
        continue

    if variances_data_started and re.match(r"\s*\d+", line):
        variances_data.append([float(num) for num in re.findall(r"-?\d+\.\d+", line)])

    if means_data_started and re.match(r"\s*\d+", line):
        means_data.append([float(num) for num in re.findall(r"-?\d+\.\d+", line)])

    if std_dev_data_started and re.match(r"\s*\d+", line):
        std_dev_data.append([float(num) for num in re.findall(r"-?\d+\.\d+", line)])

# Create DataFrames
variances_df = pd.DataFrame(variances_data)
means_df = pd.DataFrame(means_data)
std_dev_df = pd.DataFrame(std_dev_data)

# Debugging print statements
print("Variances DataFrame Shape:", variances_df.shape)
print("Means DataFrame Shape:", means_df.shape)
print("Standard Deviations DataFrame Shape:", std_dev_df.shape)


# Specify the output folder for CSV files
output_folder = "/Users/yunbo-max/Desktop/Papers/PFL_Optimiozation/Output/excel"

# Save DataFrames to CSV files with relative paths
variances_df.to_csv(
    "../PFL_Optimiozation/Output/excel/variances_data_100.csv", index=False
)
means_df.to_csv("../PFL_Optimiozation/Output/excel/means_data_100.csv", index=False)
std_dev_df.to_csv("../PFL_Optimiozation/Output/excel/std_dev_data_100.csv", index=False)
