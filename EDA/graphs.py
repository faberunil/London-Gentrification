import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np
import pandas as pd
import EDA.df_variations as dv
import json
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

############################################################################### Create a line plot of the yearly trends for each borough with summarized_df ###############################################################################
def plot_yearly_trends(data, variable):
    # Calculate the trend line by aggregating the data by year and calculating the mean for the specified variable
    yearly_trend = data.groupby('Year')[variable].mean().reset_index()
    slope, intercept, r_value, p_value, std_err = linregress(yearly_trend['Year'], yearly_trend[variable])

    # Generate x-values for the trend line, which are the unique years in the data
    x_trend = pd.Series(yearly_trend['Year'])
    y_trend = slope * x_trend + intercept

    # Create the interactive plot with Plotly Express
    fig = px.line(data, x='Year', y=variable, color='Area', title=f'Yearly {variable} for Each Borough (With Trend Line)',
                  labels={'Year': 'Year', variable: variable})
    
    # Add the trend line to the figure
    fig.add_traces(px.line(x=x_trend, y=y_trend, labels={'value': 'Trend Line'}).data[0])
    fig.data[-1].name = 'Trend Line'  # Rename the trend line trace
    fig.data[-1].line.dash = 'dash'  # Set the line style to dash

    # Update layout for better visualization
    fig.update_layout(
        legend_title_text='Borough',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20)
        )

    fig.update_layout(
        legend_title_text='Borough',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-1,  # Push the legend down
            xanchor="center",
            x=0.5  # Center the legend
            )
        )

    return fig

############################################################################### Create a line plot of the yearly rate trends for each borough with rate_change_df ###############################################################################
def plot_yearly_rate_trends(data, variable):
    # Calculate the trend line by aggregating the data by year and calculating the mean for the specified variable
    yearly_trend = data.groupby('Year')[variable].mean().reset_index()
    slope, intercept, r_value, p_value, std_err = linregress(yearly_trend['Year'], yearly_trend[variable])

    # Generate x-values for the trend line, which are the unique years in the data
    x_trend = pd.Series(yearly_trend['Year'])
    y_trend = slope * x_trend + intercept

    # Create the interactive plot with Plotly Express
    fig = px.line(data, x='Year', y=variable, color='Area', title=f'Yearly {variable} Rate Change for Each Borough (With Trend Line)',
                  labels={'Year': 'Year', variable: variable})
    
    # Add the trend line to the figure
    fig.add_traces(px.line(x=x_trend, y=y_trend, labels={'value': 'Trend Line'}).data[0])
    fig.data[-1].name = 'Trend Line'  # Rename the trend line trace
    fig.data[-1].line.dash = 'dash'  # Set the line style to dash

    # Update layout for better visualization
    fig.update_layout(
        legend_title_text='Borough',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20)
        )

    fig.update_layout(
        legend_title_text='Borough',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-1,  # Push the legend down
            xanchor="center",
            x=0.5  # Center the legend
            )
        )

    return fig

############################################################################### Create a scatter plot of the variable vs. variable for each borough with a trend line ###############################################################################
def plot_variable_vs_variable_scatter(data, var_x, var_y):
    # Calculate the trend line
    slope, intercept, r_value, p_value, std_err = linregress(data[var_x], data[var_y])

    # Create the seaborn scatter plot
    plt.figure(figsize=(14, 14))
    scatter_plot = sns.scatterplot(data=data, x=var_x, y=var_y, hue='Area', legend=True, palette='tab10')

    # Generate x-values for the trend line
    x = np.linspace(data[var_x].min(), data[var_x].max(), 100)
    # Calculate the corresponding y-values using the slope and intercept
    y = slope * x + intercept
    plt.plot(x, y, label='Trend Line', linestyle='--', color='black')

    # Customize legend
    plt.legend(title='Borough', loc='lower center', bbox_to_anchor=(0.5, -0.95), ncol=4, borderaxespad=1)

    # Add title and labels
    plt.title(f'{var_y} vs. {var_x} for Each Borough (With Trend Line)')
    plt.xlabel(var_x)
    plt.ylabel(var_y)
    # Make Y-axis start at 0
    plt.ylim(0, None)
    # Use tight_layout to automatically adjust subplot params so the plot fits into the figure area
    plt.tight_layout()

    # Display the plot
    plt.show()

############################################################################### Create a heatmap of the correlation matrix of the variables
###############################################################################
def plot_correlation_matrix(dv):
    # Calculate the correlation matrix
    corr_matrix = dv.summarized_df.drop(['Area', 'Year'], axis=1).corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(12, 10))  # Adjust figure size as needed
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix of Variables')
    # Rotate labels on the x-axis
    plt.xticks(rotation=45, ha='right')
    # Rotate labels on the y-axis
    plt.yticks(rotation=0)
    # Use tight_layout to automatically adjust subplot params so the plot fits into the figure area
    plt.tight_layout()

###############################################################################
# Map of London with Summarised Data ##########################################
###############################################################################
def plot_map(data, variable):
    # Load GeoJSON
    with open('EDA/londonboroughs.geojson') as f:
        geojson_data = json.load(f)

    # Define the color range for the variable to include the entire range across all years
    color_scale_min = data[variable].min()
    color_scale_max = data[variable].max()

    # Create frames for each year
    frames = []
    for year in range(2010, 2021):
        year_data = data[data['Year'] == year]
        frame_data = go.Choroplethmapbox(
            geojson=geojson_data,
            locations=year_data['Area'], 
            z=year_data[variable],
            featureidkey="properties.name",
            colorscale="Viridis",
            zmin=color_scale_min,
            zmax=color_scale_max,
            marker_opacity=0.5,
            marker_line_width=0
        )
        frames.append(go.Frame(data=[frame_data], name=str(year)))

    # Initial layout
    initial_layout = go.Layout(
        mapbox_style="carto-positron",
        mapbox_zoom=8.5,
        mapbox_center={"lat": 51.4974, "lon": -0.1278},
        margin={"r":0, "t":0, "l":0, "b":0},
        coloraxis=dict(colorscale="Viridis", cmin=color_scale_min, cmax=color_scale_max),
        coloraxis_colorbar=dict(title=variable)
    )

    # Create the initial figure with the layout settings
    fig = go.Figure(data=[frames[0].data[0]], layout=initial_layout, frames=frames)

    # Update layout with slider
    fig.update_layout(
        sliders=[{
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 100, "redraw": True}}],
                 "label": str(year), "method": "animate"}
                for year, f in zip(range(2010, 2021), frames)
            ],
            "transition": {"duration": 100},
            "x": 0.05, "y": 0, "currentvalue": {"prefix": "Year: "}
        }],
        title_text=f"London Boroughs {variable} 2010 - 2021)"
    )

    return fig