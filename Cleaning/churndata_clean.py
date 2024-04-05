import pandas as pd

def churn_clean(churn_csv, codes_csv, borough_txt):
    """
    Clean and preprocess churn data by merging, filtering, and calculating means.
    """
    # Read the first CSV file into a DataFrame
    churn_df = pd.read_csv(churn_csv)

    # Read the second CSV file into a DataFrame
    codes_df = pd.read_csv(codes_csv)

    codes_df.rename(columns={'LSOA04CD': 'area'}, inplace=True)

    # Merge the two DataFrames on the common column "Area"
    churn_updated = pd.merge(churn_df, codes_df, on='area', how='inner')

    # Replace the values in column "Area" with the values from column "LSOA04NM"
    churn_updated['area'] = churn_updated['LSOA04NM']

    # Drop duplicates if any
    churn_updated.drop_duplicates(inplace=True)

    # Drop the 'LSOA04NM' and 'FID' columns
    churn_updated.drop(columns=['LSOA04NM', 'FID'], inplace=True)

    # Split the strings in the "Area" column from the end and keep only the first part
    churn_updated['area'] = churn_updated['area'].str.rsplit(n=1).str[0]

    # Read borough names from the text file
    with open(borough_txt, 'r') as file:
        borough_names = [line.strip() for line in file]

    # Filter the merged DataFrame to keep only rows with Area values that match the borough names
    churn_updated = churn_updated[churn_updated['area'].isin(borough_names)]

    # Extract only the numeric part from column names (excluding 'Area')
    churn_updated.columns = [col.split('chn')[-1] if col != 'area' else col for col in churn_updated.columns]

    # Group by 'Area' and calculate the mean for each group
    churn_updated = churn_updated.groupby('area').mean().reset_index()
    
    churn_updated = churn_updated.melt(id_vars='area', var_name='Year', value_name='Churn Rate')

    # area rename and Year type change: helpful for future merging
    churn_updated.rename(columns={'area': 'Area'}, inplace=True)
    churn_updated['Year'] = churn_updated['Year'].astype(int)

    return churn_updated