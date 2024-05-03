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
merged2010_df = mcd.merged2010_df
# Convert all columns to numeric
for col in merged2010_df.columns:
    if col != 'Area':  # Skip the 'Area' column
        # convert to numeric, coercing errors to NaN
        merged2010_df[col] = pd.to_numeric(merged2010_df[col], errors='coerce')

# Rename Price to House Price
merged2010_df.rename(columns={'Price': 'House Price'}, inplace=True)

# Remove City of London Borough as lots of incomplete data
merged2010_df = merged2010_df[merged2010_df['Area'] != 'City of London']

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
summarized_df = merged2010_df[['Area', 'Year']].copy()

# Add the summaries directly to summarized_df without modifying merged2010_df
summarized_df[['Churn Rate', 'House Price', 'Earnings', 'Population']] = merged2010_df[['Churn Rate', 'House Price', 'Earnings', 'Population']]
summarized_df['White Collar Jobs'] = merged2010_df[white_collar_cols].sum(axis=1)
summarized_df['Blue Collar Jobs'] = merged2010_df[blue_collar_cols].sum(axis=1)
summarized_df['Service Sector Jobs'] = merged2010_df[service_sector_cols].sum(axis=1)
summarized_df['Violent Crimes'] = merged2010_df[violent_crimes_cols].sum(axis=1)  # First 3 are violent crimes
summarized_df['Property Crimes'] = merged2010_df[property_crimes_cols].sum(axis=1)  # Next 4 are property crimes
summarized_df['Other Offences'] = merged2010_df[other_offences_cols].sum(axis=1)  # Remaining are other offences
summarized_df['Percentage White'] = (merged2010_df['White'] / merged2010_df['Total Population']) * 100
summarized_df['Total Development'] = merged2010_df[development_cols].sum(axis=1)

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
# pop_adj_df ##################################################################
###############################################################################
# Create a new DataFrame that includes just 'Area' and 'Year' initially
pop_adj_df = summarized_df[['Area', 'Year']].copy()

# Calculate the adjusted population for each column (except 'Area' and 'Year') and add it to the new DataFrame
for col in summarized_df.columns.difference(['Area', 'Year']):
    # Calculate the adjusted population and fill NaN values with 0 in the new DataFrame
    pop_adj_df[f'{col} Pop Adjusted'] = summarized_df[col] / summarized_df['Population'] * 100

# Drop columns that arent necessary
pop_adj_df = pop_adj_df.drop(columns=['Churn Rate Pop Adjusted', 'Earnings Pop Adjusted', 'Percentage White Pop Adjusted', 'Population Pop Adjusted', 'House Price Pop Adjusted'])

###############################################################################
# normalized_df ###############################################################
###############################################################################
# Create a new DataFrame that includes just 'Area' and 'Year' initially
normalized_df = summarized_df[['Area', 'Year', 'Churn Rate', 'House Price', 'Earnings', 'Percentage White']].copy()

# Add columns from pop_adj_df to normalized_df
normalized_df = normalized_df.merge(pop_adj_df, on=['Area', 'Year'])

# Normalize the data for each column (except 'Area' and 'Year')
for col in normalized_df.columns.difference(['Area', 'Year']):
    # Calculate the normalized value and fill NaN values with 0 in the new DataFrame
    normalized_df[f'{col} Normalized'] = (normalized_df[col] - normalized_df[col].mean()) / normalized_df[col].std()
    # Drop the original column
    normalized_df = normalized_df.drop(columns=[col])

###############################################################################
# gentrification_df ###########################################################
###############################################################################
# Merge the necessary columns from each DataFrame into one DataFrame
gentrification_df = summarized_df[['Area', 'Year', 'Churn Rate']].merge(
    rate_change_df[['Area', 'Year', 'House Price Rate Change', 'Earnings Rate Change', 'White Collar Jobs Rate Change']],
    on=['Area', 'Year']
)

# Calculate the gentrification index
gentrification_df['gentrification_index'] = (
    0.75 * gentrification_df['Churn Rate'] +
    gentrification_df['House Price Rate Change'] +
    gentrification_df['Earnings Rate Change'] +
    gentrification_df['White Collar Jobs Rate Change']
)

# Set gentrification index to NaN for year 2010
gentrification_df.loc[gentrification_df['Year'] == 2010, 'gentrification_index'] = pd.NA

# Drop the now-unused rate change columns if they are not needed
gentrification_df.drop(columns=['Churn Rate', 'House Price Rate Change', 'Earnings Rate Change', 'White Collar Jobs Rate Change'], inplace=True)

# add gentrification_index col to summarized_df, rate_change_df, pop_adj_df, normalized_df
summarized_df = summarized_df.merge(gentrification_df[['Area', 'Year', 'gentrification_index']], on=['Area', 'Year'])
rate_change_df = rate_change_df.merge(gentrification_df[['Area', 'Year', 'gentrification_index']], on=['Area', 'Year'])
pop_adj_df = pop_adj_df.merge(gentrification_df[['Area', 'Year', 'gentrification_index']], on=['Area', 'Year'])
normalized_df = normalized_df.merge(gentrification_df[['Area', 'Year', 'gentrification_index']], on=['Area', 'Year'])

# Using bins make a new column named Classification in a new df named summarized_final to show the level of gentrification
# Define bins and labels for the classification
bins = [0, 0.2, 0.4, 0.6, 1]
labels = ['Low/No Gentrification', 'Mild Gentrification', 'High Gentrification', 'Extreme Gentrification']

summarized_final = summarized_df.copy()
summarized_final['Classification'] = pd.cut(summarized_final['gentrification_index'], bins=bins, labels=labels)