import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import EDA.graphs as graphs
import EDA.df_variations as dv

st.set_option('deprecation.showPyplotGlobalUse', False)

# Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    'Select a Page',
    ('Introduction', 'Dataframes','Exploratory Data Analysis', 'Map Visualizations', 'Modeling', 'Test Your Data')
)

# Page Title
st.title("London Gentrification Data Visualization Dashboard")

if option == 'Introduction':
    st.header("London Gentrification Data Visualization Dashboard")
    st.write("Please select a page on the left sidebar to view different data visualizations.")


elif option == 'Dataframes':
    st.header("Variations of Dataframes")
    st.subheader("The following dataframes are available for analysis:")
    st.write("1. Summarized Dataframe")
    st.dataframe(dv.summarized_df)
    st.write("2. Rate Change Dataframe")
    st.dataframe(dv.rate_change_df)
    st.write("3. Normalized Dataframe")
    st.dataframe(dv.normalized_df)
    st.subheader("Distribution of Data")

    # Display the number of unique areas
    num_unique_areas = dv.summarized_df['Area'].nunique()
    st.write(f"Number of unique areas: {num_unique_areas}")

    st.subheader("Distribution of Data")
    
    variables = dv.summarized_df.columns.drop(['Area', 'Year'])  # Drop 'Area' and 'Year' columns from selection
    # Ask user to select a variable to view distribution
    vars = st.selectbox("Select a variable to view distribution", variables)
    
    # Interactive Histogram using Plotly
    st.subheader('Interactive Histogram')
    hist_fig = px.histogram(dv.summarized_df, x=vars, nbins=30, title='Histogram')
    hist_fig.update_layout(bargap=0.1)
    st.plotly_chart(hist_fig, use_container_width=True)
    
    # Interactive Boxplot using Plotly
    st.subheader('Interactive Boxplot')
    box_fig = px.box(dv.summarized_df, y=vars, title='Boxplot')
    st.plotly_chart(box_fig, use_container_width=True)

elif option == 'Exploratory Data Analysis':
    st.header("Exploratory Data Analysis")
    st.subheader("Time Series Analysis")
    st.write("The following plots show the yearly trends for each borough in a total figures format and also an annual rate change format.")
    st.write("Total figure Time Series")
    st.markdown("<br>", unsafe_allow_html=True)
    # Create a dropdown for selecting the variable to plot for rate change
    svariables = dv.summarized_df.columns.drop(['Area', 'Year'])  # Drop 'Area' and 'Year' columns from selection
    rvariables = dv.rate_change_df.columns.drop(['Area', 'Year'])
    
    sum_variable = st.selectbox("Select a variable for General Time Series Analysis", svariables)

    fig_sum = graphs.plot_yearly_trends(dv.summarized_df, sum_variable)
    st.plotly_chart(fig_sum, use_container_width=True)

    st.write("Rate Change Time Series")
    st.markdown("<br>", unsafe_allow_html=True)

    rate_variable = st.selectbox("Select a variable for Rate Change Time Series Analysis", rvariables)

    fig_rate = graphs.plot_yearly_rate_trends(dv.rate_change_df, rate_variable)
    st.plotly_chart(fig_rate, use_container_width=True)

    st.subheader("Multivariate Scatter Plot Analysis")
    st.write("Multivariate Scatter Plots from the summarized dataframe. Select two variables to plot against each other.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Create a dropdown for selecting the variables to plot
    var1 = st.selectbox("Select 1st variable for Scatter Plot", svariables)
    var2 = st.selectbox("Select 2nd variable for Scatter Plot", svariables, index=1)
    st.pyplot(graphs.plot_variable_vs_variable_scatter(dv.summarized_df, var2 , var1))

    st.subheader("Correlation Matrix")
    st.write("The correlation matrix shows the relationship between each pair of variables in the summarized dataframe.")
    st.markdown("<br>", unsafe_allow_html=True)

    plot_container3 = st.empty() # Create a placeholder for the third plot
    plot_container3.pyplot(graphs.plot_correlation_matrix(dv))

elif option == 'Map Visualizations':
    st.header("Map Visualization for each variable in the Summarized Dataframe")

    # Create a dropdown for selecting the variable to plot for rate change
    svariables = dv.summarized_df.columns.drop(['Area', 'Year'])  # Drop 'Area' and 'Year' columns from selection
    sum_variable = st.selectbox("Select a variable for Map Visualization", svariables)
    # Display map plots
    st.plotly_chart(graphs.plot_map(dv.summarized_df, sum_variable), theme="streamlit", use_container_width=False)

elif option == 'Modeling':
    st.header("Modeling")
    st.write("Modeling insights will be displayed here.")

elif option == 'Test Your Data':
    st.header("Test Your Data")
    st.write("Test your data")

# Example for another navigation option
# if option == 'Another Section':
#     st.header("Another Section")
#     # More content here

# Adding a footer
st.markdown("---")
st.markdown("© Faber Bickerstaffe 2024 Data Visualization App")

# This is a simple Streamlit app that displays "Hello World" on the screen.
# To run the app, open a terminal and run the following command:
# python -m streamlit run streamlit_app.py
