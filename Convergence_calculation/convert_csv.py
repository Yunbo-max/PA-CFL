import pandas as pd
import re

# Read the log file
with open("./Parallel_computation/optimised_fl_ray_10_11.log", "r") as file:
    log_data = file.readlines()

# Extract data from the log
variances_data = []
means_data = []
std_dev_data = []

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

# Save DataFrames to CSV files
variances_df.to_csv("variances_data.csv", index=False)
means_df.to_csv("means_data.csv", index=False)
std_dev_df.to_csv("std_dev_data.csv", index=False)
