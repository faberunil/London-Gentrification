import sys
import os

# Get the path to the parent directory (main folder)
parent_directory = os.path.abspath('.')

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import pandas as pd
from Cleaning.churndata_clean import churn_clean
from Cleaning.house_price_clean_pivot import process_house_prices_data
from Cleaning.jobs_market_pivot import process_jobs_data
from Cleaning.crime_clean_pivot import clean_crime_data
from Cleaning.earning_clean_pivot import clean_earnings_data
from Cleaning.demographic_clean_merge import merge_ethnicity_data
from Cleaning.develop_merge_clean import merge_and_clean_borough_developments

# Call each cleaning function
borough_rename_df = churn_clean('Data/hh_churn.csv', 'Data/uk_areacodes.csv', 'Data/Boroughs.txt')
house_prices_df = process_house_prices_data('Data/UK House price index.xlsx', 'Data/Boroughs.txt')
jobs_data_df = process_jobs_data('Data/2023 borough by sector.csv')
crime_data_df = clean_crime_data('Data/MPS Borough Level Crime (Historical).csv', 'Data/Boroughs.txt')
earnings_data_df = clean_earnings_data('Data/earnings-residence-borough.xlsx', 'Data/Boroughs.txt')
demographics_data_df = merge_ethnicity_data()
development_data_df = merge_and_clean_borough_developments()

# Create a list of DataFrames to merge
dfs_to_merge = [borough_rename_df, house_prices_df, jobs_data_df, crime_data_df, earnings_data_df, demographics_data_df, development_data_df]

# Define the columns to merge on
merge_columns = ['Year', 'Area']

# Merge the DataFrames
raw_merged_df = dfs_to_merge[0]  # Initialize with the first DataFrame
for df in dfs_to_merge[1:]:
    raw_merged_df = pd.merge(raw_merged_df, df, on=merge_columns, how='outer')

merged2010_df = raw_merged_df[(raw_merged_df['Year'] >= 2010) & (raw_merged_df['Year'] <= 2020)]