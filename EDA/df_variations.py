import sys
import os

# Get the path to the parent directory (main folder)
parent_directory = os.path.abspath('.')

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import Cleaning.merge_clean_data as mcd
import pandas as pd
# Import the file that contains the merged_df variable

# Access the merged_df variable using the module name
raw_merged2010_df = mcd.raw_merged2010_df
# Convert all columns to numeric
for col in raw_merged2010_df.columns:
    if col != 'Area':  # Skip the 'Area' column
        # convert to numeric, coercing errors to NaN
        raw_merged2010_df[col] = pd.to_numeric(raw_merged2010_df[col], errors='coerce')

# Rename Price to House Price
raw_merged2010_df.rename(columns={'Price': 'House Price'}, inplace=True)

# Remove City of London Borough as lots of incomplete data
raw_merged2010_df = raw_merged2010_df[raw_merged2010_df['Area'] != 'City of London']

###############################################################################
# Summarised_df ################################################################
###############################################################################
# Define the columns for each category
white_collar_cols = ['Professional, Real Estate, Scientific and technical activities', 'Financial and insurance activities', 'Information and Communication']
blue_collar_cols = ['Construction', 'Manufacturing', 'Transportation and Storage']
service_sector_cols = ['Accomodation and food service activities', 'Retail', 'Other services', 'Administrative and support service activities', 'Health', 'Education']
violent_crimes_cols = ['Robbery', 'Violence Against the Person', 'Sexual Offences']
property_crimes_cols = ['Burglary', 'Theft', 'Vehicle Offences', 'Arson and Criminal Damage']
other_offences_cols = ['Drug Offences', 'Public Order Offences', 'Possession of Weapons', 'Miscellaneous Crimes Against Society', 'Historical Fraud and Forgery']
development_cols = ['Conversion_count', 'Major all other major developments_count', 'Minor dwellings_count', 'New Build_count', 'Major dwellings_count', 'Minor retail and service_count', 'Minor Offices-R and D-light industry_count', 'Major general industry-storage-warehousing_count']

# Create summarized_df with Area and Year
summarized_df = raw_merged2010_df[['Area', 'Year']].copy()

# Add the summaries directly to summarized_df without modifying raw_merged2010_df
summarized_df[['Churn Rate', 'House Price', 'Earnings']] = raw_merged2010_df[['Churn Rate', 'House Price', 'Earnings']]
summarized_df['White Collar Jobs'] = raw_merged2010_df[white_collar_cols].sum(axis=1)
summarized_df['Blue Collar Jobs'] = raw_merged2010_df[blue_collar_cols].sum(axis=1)
summarized_df['Service Sector Jobs'] = raw_merged2010_df[service_sector_cols].sum(axis=1)
summarized_df['Violent Crimes'] = raw_merged2010_df[violent_crimes_cols].sum(axis=1)  # First 3 are violent crimes
summarized_df['Property Crimes'] = raw_merged2010_df[property_crimes_cols].sum(axis=1)  # Next 4 are property crimes
summarized_df['Other Offences'] = raw_merged2010_df[other_offences_cols].sum(axis=1)  # Remaining are other offences
summarized_df['Percentage White'] = (raw_merged2010_df['White'] / raw_merged2010_df['Total Population']) * 100
summarized_df['Total Development'] = raw_merged2010_df[development_cols].sum(axis=1)

# Now summarized_df contains 'Area', 'Year', and the summaries for each category

###############################################################################
# rate_change_df ##############################################################
###############################################################################
# Create a new DataFrame that includes just 'Area' and 'Year' initially
rate_change_df = summarized_df[['Area', 'Year']].copy()

# Calculate the yearly rate change for each column (except 'Area' and 'Year') and add it to the new DataFrame
for col in summarized_df.columns.difference(['Area', 'Year']):
    # Calculate the percentage change and fill NaN values with 0 in the new DataFrame
    rate_change_df[f'{col} Rate Change'] = summarized_df.groupby('Area')[col].pct_change().fillna(0)

# Now rate_change_df contains 'Area', 'Year', and the rate change for each summary category

###############################################################################
# normalized_df ###############################################################
###############################################################################
# Create a new DataFrame that includes just 'Area' and 'Year' initially
normalized_df = summarized_df[['Area', 'Year']].copy()

# Normalize the data for each column (except 'Area' and 'Year') and add it to the new DataFrame
for col in summarized_df.columns.difference(['Area', 'Year']):
    # Calculate the z-score for each value and fill NaN values with 0 in the new DataFrame
    normalized_df[f'{col} Normalized'] = (summarized_df[col] - summarized_df[col].mean()) / summarized_df[col].std()