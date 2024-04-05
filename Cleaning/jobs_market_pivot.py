import pandas as pd

def process_jobs_data(csv_file):
    """
    File containing job market data, filter the data for years 2010 and after,
    pivot the DataFrame to have each sector as its own column.
    for Data/2023 borough by sector.csv
    """
    # Read the CSV file into a DataFrame
    jobs_df = pd.read_csv(csv_file)

    # Filter rows to include only data from 2010 and after
    jobs_df = jobs_df[jobs_df['Year'] >= 2010]
    
    # Pivot the DataFrame to have each sector as its own column
    pivoted_jobs_df = jobs_df.pivot_table(index=['Borough', 'Year'], columns='Sector', values='Employee jobs', aggfunc='sum', fill_value=0)

    # Reset the index to make 'Borough' and 'Year' regular columns
    pivoted_jobs_df.reset_index(inplace=True)

    # Optional: Rename the columns for clarity and mergeability
    pivoted_jobs_df.rename(columns={'Borough': 'Area'}, inplace=True)
    pivoted_jobs_df.columns.name = None  # Remove the name of the index

    return pivoted_jobs_df