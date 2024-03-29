import pandas as pd
import re

def merge_and_clean_borough_developments():
    """
    Merge and clean borough development data from CSV files for the years 2010 to 2020.
    """
    # Define a list to hold the DataFrame objects
    develop_count_dfs = []

    # Define a list of file paths for years from 2010 to 2020
    file_paths = ['Data\Borough_developments_{}.csv'.format(year) for year in range(2010, 2021)]

    # Iterate over the file paths and read each CSV file
    for file_path in file_paths:
        # Read the CSV file into a DataFrame
        development_df = pd.read_csv(file_path)

        # Preprocess the "borough" column
        development_df['borough'] = development_df['borough'].str.replace('&', 'and')  # Replace "&" with "and"
        development_df['borough'] = development_df['borough'].str.replace('London Borough of ', '')  # Remove "London Borough of "
        development_df['borough'] = development_df['borough'].str.replace('Royal Borough of ', '')  # Remove "Royal Borough of "
        development_df['borough'] = development_df['borough'].str.replace(' \(LA Code\)', '')  # Change case to "Kingston Upon Thames"
        development_df['borough'] = development_df['borough'].str.replace('Kingston upon Thames', 'Kingston')  # Change case to "Kingston Upon Thames step 1"
        development_df['borough'] = development_df['borough'].str.replace('Kingston', 'Kingston upon Thames')  # Change case to "Kingston Upon Thames step 2"

        # Pivot the DataFrame to count the occurrences of each development type for each borough
        pivot_df = development_df.pivot_table(index='borough', columns='development_type', aggfunc='size', fill_value=0)

        # Reset the index to make 'borough' a column again
        pivot_df.reset_index(inplace=True)

        # Rename the columns for clarity
        pivot_df.columns.name = None  # Remove the name of the index
        pivot_df.columns = ['Borough'] + ['{}_count'.format(col) for col in pivot_df.columns[1:]]

        # Add a 'Year' column with the year extracted from the file name
        year = file_path.split('_')[-1].split('.')[0]  # Extract the year from the file name
        pivot_df.insert(1, 'Year', year)  # Insert the 'Year' column as the second column

        # Append the DataFrame to the develop_count_dfs list
        develop_count_dfs.append(pivot_df)

    # Merge all DataFrames in the develop_count_dfs list into a single DataFrame
    develop_merged = pd.concat(develop_count_dfs, ignore_index=True)

    # Replace NaN values with 0
    develop_merged.fillna(0, inplace=True)

    # Save the merged DataFrame to a CSV file
    develop_merged.to_csv('Borough_developments_counts.csv', index=False)

    return develop_merged