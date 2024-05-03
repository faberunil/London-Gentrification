import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import EDA.graphs as graphs
import EDA.df_variations as dv
import Modeling.GB_predict as gb

# Load results_df csv from Models
results_df = pd.read_csv('Models/results_df.csv')

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
    st.subheader("1. Summarized Dataframe")
    st.write("The Summarized dataframe shows the total figures for each variable in the London data.")
    st.dataframe(dv.summarized_df)
    st.subheader("2. Rate Change Dataframe")
    st.write("The Rate Change dataframe shows the annual rate of change for each variable in the summarized dataframe. Values begin in 2011.")
    st.dataframe(dv.rate_change_df)
    st.subheader("3. Population Adjusted Dataframe")
    st.write("The Population Adjusted dataframe shows the population adjusted values for each variable in the summarized dataframe.")
    st.dataframe(dv.pop_adj_df)
    st.subheader("4. Normalized Dataframe")
    st.write("The Normalized dataframe shows the normalized values for each variable in the summarized dataframe.")
    st.dataframe(dv.normalized_df)
    
    st.header("Distribution of Data")

    # Display the number of unique areas
    num_unique_areas = dv.summarized_df['Area'].nunique()
    st.write(f"There are unique Boroughs: {num_unique_areas} in London")

    st.subheader("Distribution of Data")
    st.write("The following plots show the distribution of the data in the summarized dataframe.")
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

    st.write("The following plots show the distribution of the data in the Population Adjusted dataframe.")
    pvariables = dv.pop_adj_df.columns.drop(['Area', 'Year'])  # Drop 'Area' and 'Year' columns from selection
    # Ask user to select a variable to view distribution
    pvars = st.selectbox("Select a variable to view distribution", pvariables)
    
    # Interactive Histogram using Plotly
    st.subheader('Interactive Histogram Pop Adjusted')
    hist_fig = px.histogram(dv.pop_adj_df, x=pvars, nbins=30, title='Histogram')
    hist_fig.update_layout(bargap=0.1)
    st.plotly_chart(hist_fig, use_container_width=True)
    
    # Interactive Boxplot using Plotly
    st.subheader('Interactive Boxplot Pop Adjusted')
    box_fig = px.box(dv.pop_adj_df, y=pvars, title='Boxplot')
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
    pvariables = dv.pop_adj_df.columns.drop(['Area', 'Year'])

    sum_variable = st.selectbox("Select a variable for General Time Series Analysis", svariables)

    fig_sum = graphs.plot_yearly_trends(dv.summarized_df, sum_variable)
    st.plotly_chart(fig_sum, use_container_width=True)

    st.write("Rate Change Time Series")
    st.markdown("<br>", unsafe_allow_html=True)

    rate_variable = st.selectbox("Select a variable for Rate Change Time Series Analysis", rvariables)

    fig_rate = graphs.plot_yearly_rate_trends(dv.rate_change_df, rate_variable)
    st.plotly_chart(fig_rate, use_container_width=True)

    popadj_variable = st.selectbox("Select a variable for Population Adjusted Time Series Analysis", pvariables)

    fig_popadj = graphs.plot_yearly_trends_pop_adj(dv.pop_adj_df, popadj_variable)
    st.plotly_chart(fig_popadj, use_container_width=True)

    st.subheader("Multivariate Scatter Plot Analysis")
    st.write("Multivariate Scatter Plots from the summarized dataframe. Select two variables to plot against each other.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Create a dropdown for selecting the variables to plot
    var1 = st.selectbox("Select 1st variable for Scatter Plot", svariables)
    var2 = st.selectbox("Select 2nd variable for Scatter Plot", svariables, index=1)
    st.pyplot(graphs.plot_variable_vs_variable_scatter(dv.summarized_df, var2 , var1))

    st.subheader("Correlation Matrix")
    st.write("This correlation matrix shows the relationship between each pair of variables in the summarized dataframe.")
    st.markdown("<br>", unsafe_allow_html=True)

    plot_container3 = st.empty() # Create a placeholder for the third plot
    plot_container3.pyplot(graphs.plot_correlation_matrix(dv.summarized_df))

    st.write("This correlation matrix shows the relationship between each pair of variables in the Normalized dataframe.")
    st.markdown("<br>", unsafe_allow_html=True)

    plot_container3 = st.empty() # Create a placeholder for the third plot
    plot_container3.pyplot(graphs.plot_correlation_matrix(dv.normalized_df))

elif option == 'Map Visualizations':
    st.header("Map Visualization for each variable in the Summarized Dataframe")

    # Create a dropdown for selecting the variable to plot for rate change
    svariables = dv.summarized_df.columns.drop(['Area', 'Year'])  # Drop 'Area' and 'Year' columns from selection
    sum_variable = st.selectbox("Select a variable for Map Visualization", svariables)
    # Display map plots
    st.plotly_chart(graphs.plot_map(dv.summarized_df, sum_variable), theme="streamlit", use_container_width=False)

    st.header("Map Visualization for each variable in the Population Adjusted Dataframe")

    # Create a dropdown for selecting the variable to plot for rate change
    pvariables = dv.pop_adj_df.columns.drop(['Area', 'Year'])  # Drop 'Area' and 'Year' columns from selection
    pop_variable = st.selectbox("Select a variable for Map Visualization", pvariables)
    # Display map plots
    st.plotly_chart(graphs.plot_map(dv.pop_adj_df, pop_variable), theme="streamlit", use_container_width=False)

if option == 'Modeling':
    st.header("Modeling")
    st.header("Classification Models")
    st.write("After trialling 5 classifying models we found that Gradient Boosting has the highest accuracy for our London Summarized data, this will be used to predict the future gentrification of London 2030. Random Forest was found to be best at classifiying the Rate Change data. This is more useful for worldwide application of data as the data does not have to be London specific.")

    st.write("The following table shows the results of the 5 classification models.")

    # Let users filter by DataFrame
    df_option = st.selectbox(
        'Select DataFrame to filter results:',
        results_df['DataFrame'].unique()
    )

    # Filter the DataFrame based on selection
    filtered_df = results_df[results_df['DataFrame'] == df_option]

    # Display the filtered DataFrame
    st.dataframe(filtered_df)

    st.header("Predictions for 2030")
    st.write("The following table shows the predictions for 2030 using the Gradient Boosting model.")

    # Display the predictions for 2030
    st.dataframe(gb.gru_predictions_final)

    # Display a map of the predictions for 2030
    st.plotly_chart(graphs.plot_map_class(gb.gru_predictions_final, 'Classifications'), theme="streamlit", use_container_width=False)

elif option == 'Test Your Data':
    st.header("Test Your Data")
    st.write("Now you have the opportunity to test your data using the Random Forest model. Input your yearly rate change data and the model will predict the classification for you.")

    # Create a form for users to input data
    form = st.form(key='input_form')
    # Create input fields for each feature
    st.write("Input your data:")

# Adding a footer
st.markdown("---")
st.markdown("© Faber Bickerstaffe 2024 Data Visualization App")

# To run the app, open a terminal and run the following command:
# python -m streamlit run streamlit_app.py