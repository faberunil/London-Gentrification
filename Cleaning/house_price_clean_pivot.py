import pandas as pd

def process_house_prices_data(file_path, boroughs_file_path):
    """
    Clean and pivot the UK house price data.
    """

    # Read the house prices data from the Excel file
    hprice_df = pd.read_excel(file_path, sheet_name=str('Average price'))

    # Drop the first row
    hprice_df.drop([0], inplace=True)

    # Rename the 'Unnamed: 0' column to 'Year'
    hprice_df.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)

    # Extract the year from the dates in the 'Year' column
    hprice_df['Year'] = pd.to_datetime(hprice_df['Year']).dt.year

    # Group by year and calculate the average house prices for each year
    yearly_av_hprice_df = hprice_df.groupby('Year').mean()

    # Filter for the years between 2010 and 2021
    average_prices_2010_to_2021 = yearly_av_hprice_df.loc[2010:2021]

    # Reset index to make 'Year' a regular column
    average_prices_2010_to_2021.reset_index(inplace=True)

    # Preprocess the "Area" column
    average_prices_2010_to_2021.columns = average_prices_2010_to_2021.columns.str.replace('&', 'and')  # Replace "&" with "and"

    # Read the list of boroughs
    with open(boroughs_file_path, 'r') as file:
        boroughs_list = [line.strip() for line in file]

    # Filter the DataFrame to keep only the borough columns
    borough_columns = [col for col in average_prices_2010_to_2021.columns if col in boroughs_list]
    borough_prices_df = average_prices_2010_to_2021[['Year'] + borough_columns]

    # Melt the DataFrame to pivot the borough columns into a single 'Area' column
    melted_df = borough_prices_df.melt(id_vars=['Year'], var_name='Area', value_name='Price')

    return melted_df