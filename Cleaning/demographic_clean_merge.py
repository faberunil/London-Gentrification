import pandas as pd

def merge_ethnicity_data():
    """
    Merge ethnicity data from multiple Excel tabs into a single DataFrame.
    Correct the column names and filter the data for the desired boroughs, also add a 'Year' column.
    """
    # Initialize an empty DataFrame to store the merged data
    ethnicity_df = pd.DataFrame()

    # Read data from each tab and merge into the main DataFrame
    for year in range(2012, 2021):
        # Read data from the current tab
        df = pd.read_excel('Data\ethnic-groups-by-borough.xls', sheet_name=str(year))

        # Drop the 'Code' column and all columns after 7th column
        df.drop(columns=['Code'] + list(df.columns[7:]), inplace=True)

        # Drop the first two rows as they contain non-numeric data
        df.drop([0, 1], inplace=True)

        # Rename the columns to match the desired format
        df.columns = ['Area', 'White', 'Asian', 'Black', 'Mixed/Other', 'Total Population']

        # Add a 'Year' column
        df['Year'] = year

        # Filter the DataFrame to keep only the area values that are in the 'boroughs.txt' list
        with open('Data\Boroughs.txt', 'r') as file:
            boroughs_list = [line.strip() for line in file]
        df = df[df['Area'].isin(boroughs_list)]

        # Append the current DataFrame to the main DataFrame
        ethnicity_df = pd.concat([ethnicity_df, df], ignore_index=True)

    return ethnicity_df