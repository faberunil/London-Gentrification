import pandas as pd

def clean_crime_data(crime_csv_path, boroughs_txt_path):
    """
    Clean the crime data CSV file and pivot it to a format suitable for analysis.
    """

    # Read the crime data from the CSV file
    crime_df = pd.read_csv(crime_csv_path)

    # Drop the 'MinorText' column
    crime_df.drop(columns=['MinorText'], inplace=True)

    # Rename the 'LookUp_BoroughName' column to 'Area'
    crime_df.rename(columns={'LookUp_BoroughName': 'Area'}, inplace=True)

    # Extract the year from the column names
    years = crime_df.columns[3:].str[:4]

    # Filter the DataFrame to keep only the areas that match the borough text list
    with open(boroughs_txt_path, 'r') as file:
        boroughs_list = [line.strip() for line in file]
    crime_df = crime_df[crime_df['Area'].isin(boroughs_list)]

    # Initialize an empty DataFrame to store the results
    clean_crime_df = pd.DataFrame()

    # Iterate over each unique year
    for year in years.unique():
        # Select columns for the current year
        year_cols = [col for col in crime_df.columns if year in col]

        # Create a DataFrame for the current year
        year_df = crime_df[['Area', 'MajorText'] + year_cols].copy()

        # Sum the values for each combination of Area and MajorText
        year_df['Year'] = year
        year_df['Value'] = year_df[year_cols].sum(axis=1)
        clean_crime_df = pd.concat([clean_crime_df, year_df[['Area', 'MajorText', 'Year', 'Value']]])

    # Remove the 2022 data
    clean_crime_df = clean_crime_df[clean_crime_df['Year'] != '2022']

    # Convert the 'Year' column to integer type
    clean_crime_df['Year'] = clean_crime_df['Year'].astype(int)

    # Multiply the values in the '2010' column by 4/3 (as only 9 months of data are available)
    clean_crime_df.loc[(clean_crime_df['Year'] == '2010'), 'Value'] *= 4/3

    # Round the values in the '2010' column to have no decimals
    clean_crime_df.loc[(clean_crime_df['Year'] == '2010'), 'Value'] = clean_crime_df.loc[(clean_crime_df['Year'] == '2010'), 'Value'].round(0)

    # Pivot the DataFrame
    clean_crime_pivot_df = clean_crime_df.pivot_table(index=['Area', 'Year'], columns='MajorText', values='Value', aggfunc='sum', fill_value=0).reset_index()

    return clean_crime_pivot_df