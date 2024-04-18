import pandas as pd

def clean_pop(csv_file, boroughs_file):
    # Read in the population data
    population = pd.read_csv(csv_file)

    # Drop unnecessary columns
    population = population.drop(columns=['Code', 'Source', 'Inland_Area _Hectares', 'Total_Area_Hectares', 'Population_per_hectare', 'Square_Kilometres', 'Population_per_square_kilometre'])

    # Rename 'Name' to 'Area'
    population = population.rename(columns={'Name': 'Area'})

    # Read the boroughs names from a file into a list
    with open(boroughs_file, 'r') as file:
        boroughs = [line.strip() for line in file]

    # Filter the DataFrame for areas that are in the boroughs list
    population = population[population['Area'].isin(boroughs)]

    return population