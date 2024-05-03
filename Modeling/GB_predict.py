import pandas as pd
import joblib

# Load the Gradient Boosting model
gb_model_path = 'Models/gbsum_model.pkl'
gb_model = joblib.load(gb_model_path)

# Load GRU predictions and ensure only relevant columns are present for classification
gru_predictions_path = 'Data/gru_predictions.csv'
gru_predictions = pd.read_csv(gru_predictions_path)
expected_features = ['Churn Rate', 'House Price', 'Earnings', 'Population', 'White Collar Jobs', 'Blue Collar Jobs', 'Service Sector Jobs', 'Violent Crimes', 'Property Crimes', 'Other Offences', 'Total Development']

# Preserve the Area and Year columns separately
area_data = gru_predictions[['Area', 'Year']]

# Filter out only the expected features for classification
gru_predictions_final = gru_predictions[expected_features]

# Step 3: Predict classifications using the loaded Gradient Boosting model
classification_results = gb_model.predict(gru_predictions_final)

# Add classifications to the DataFrame containing GRU predictions
gru_predictions_final['Classifications'] = classification_results

# Add the Area and Year columns back to the DataFrame
gru_predictions_final['Area'] = area_data['Area']
gru_predictions_final['Year'] = area_data['Year']

# Reorder the columns so Year is the first column and Area is the second column
gru_predictions_final = gru_predictions_final[['Year', 'Area'] + expected_features + ['Classifications']]