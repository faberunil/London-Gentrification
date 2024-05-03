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

###############################################################################
# Create a line plot of the yearly trends for each borough with pop_adj_df
###############################################################################
def plot_yearly_trends_pop_adj(data, variable):
    # Calculate the trend line by aggregating the data by year and calculating the mean for the specified variable
    yearly_trend = data.groupby('Year')[variable].mean().reset_index()
    slope, intercept, r_value, p_value, std_err = linregress(yearly_trend['Year'], yearly_trend[variable])

    # Generate x-values for the trend line, which are the unique years in the data
    x_trend = pd.Series(yearly_trend['Year'])
    y_trend = slope * x_trend + intercept

    # Create the interactive plot with Plotly Express
    fig = px.line(data, x='Year', y=variable, color='Area', title=f'Yearly {variable} Population Adjusted for Each Borough (With Trend Line)',
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
def plot_correlation_matrix(data):
    # Calculate the correlation matrix
    corr_matrix = data.drop(['Area', 'Year'], axis=1).corr()

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

###############################################################################
# Map of London with Classification ##########################################
###############################################################################
def plot_map_class(data, variable):
    # Load GeoJSON
    with open('EDA/londonboroughs.geojson') as f:
        geojson_data = json.load(f)

    # Define discrete color mapping for gentrification levels
    gentrification_levels = {
        'Low/No Gentrification': 0,
        'Mild Gentrification': 0.33,
        'High Gentrification': 0.67,
        'Extreme Gentrification': 1
    }
    
    # Map classifications to numeric values for coloring
    data['numeric_classification'] = data[variable].map(gentrification_levels)

    # Define a simple green to red colorscale
    colorscale = "RdYlGn_r"  # '_r' reverses the usual red to green scale to green to red

    # Create frames for each year
    frames = []
    for year in range(2021, 2031):
        year_data = data[data['Year'] == year]
        frame_data = go.Choroplethmapbox(
            geojson=geojson_data,
            locations=year_data['Area'],
            z=year_data['numeric_classification'],  # Use the numeric mappings
            featureidkey="properties.name",
            colorscale=colorscale,
            zmin=0,
            zmax=1,
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
        coloraxis=dict(colorscale=colorscale, cmin=0, cmax=1),
        coloraxis_colorbar=dict(title=variable, tickvals=[0, 0.33, 0.67, 1], ticktext=list(gentrification_levels.keys()))
    )

    # Create the initial figure with the layout settings
    fig = go.Figure(data=[frames[0].data[0]], layout=initial_layout, frames=frames)

    # Update layout with slider
    fig.update_layout(
        sliders=[{
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 100, "redraw": True}}],
                 "label": str(year), "method": "animate"}
                for year, f in zip(range(2021, 2031), frames)
            ],
            "transition": {"duration": 100},
            "x": 0.05, "y": 0, "currentvalue": {"prefix": "Year: "}
        }],
        title_text=f"London Boroughs Gentrification 2021 - 2030)"
    )

    return fig

###############################################################################
# Layered Bar Chart of instances of gentrification levels in each borough ######
###############################################################################
def plot_gentrification_instances(data, variable):
    # Define the color mapping for gentrification levels
    gentrification_colors = {
        'Low/No Gentrification': 'green',
        'Mild Gentrification': 'yellow',
        'High Gentrification': 'orange',
        'Extreme Gentrification': 'red'
    }

    # Create a new DataFrame to count the instances of each gentrification level in each borough
    gentrification_counts = data.groupby(['Area', variable]).size().unstack().fillna(0)

    # Create a list of colors for each gentrification level
    colors = [gentrification_colors[level] for level in gentrification_counts.columns]

    # Create the bar chart using Plotly Express
    fig = go.Figure()
    for i, level in enumerate(gentrification_counts.columns):
        fig.add_trace(go.Bar(
            x=gentrification_counts.index,
            y=gentrification_counts[level],
            name=level,
            marker_color=colors[i]
        ))

    # Update the layout for better visualization
    fig.update_layout(
        barmode='stack',
        xaxis_title='Borough',
        yaxis_title='Number of Instances',
        title=f'Instances of Gentrification Levels in Each Borough ({variable})'
    )

    return fig