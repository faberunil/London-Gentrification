import pandas as pd

def clean_earnings_data(excel_file_path, boroughs_file_path):
    """
    Clean earnings data by reading Excel file and filtering it based on boroughs and years.
    """
    # Read the Excel file
    earnings_df = pd.read_excel(excel_file_path, sheet_name='Total, weekly')

    # Drop the 'Code' column and all columns starting with 'Unnamed'
    earnings_df.drop(columns=['Code'] + list(earnings_df.filter(regex='^Unnamed').columns), inplace=True)

    # Drop the first two rows as they seem to contain non-numeric data
    earnings_df.drop([0, 1], inplace=True)

    # Convert non-numeric values to NaN and then convert the year columns to numeric
    earnings_df.iloc[:, 1:] = earnings_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Melt the dataframe to pivot all the year columns into a single column called 'Year'
    earnings_df = pd.melt(earnings_df, id_vars=['Area'], var_name='Year', value_name='Earnings')

    # Filter the DataFrame to keep only the area values that are in the 'boroughs.txt' list
    with open(boroughs_file_path, 'r') as file:
        boroughs_list = [line.strip() for line in file]

    earnings_df = earnings_df[earnings_df['Area'].isin(boroughs_list)]

    # Convert the 'Year' column to integer type
    earnings_df['Year'] = earnings_df['Year'].astype(int)

    # Filter the DataFrame to keep only years from 2010 to 2020
    earnings_df = earnings_df[(earnings_df['Year'] >= 2010) & (earnings_df['Year'] <= 2020)]

    return earnings_df