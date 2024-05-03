import sys
import os

# Get the path to the parent directory (main folder) and set up environment
parent_directory = os.path.abspath('.')
os.environ["TF_USE_LEGACY_KERAS"] = "1"
sys.path.append(parent_directory)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import EDA.df_variations as dv


# Load data
summarized_df = dv.summarized_df

feature_columns = ['Churn Rate', 'House Price', 'Earnings', 'Population', 'White Collar Jobs', 'Blue Collar Jobs', 'Service Sector Jobs', 'Violent Crimes', 'Property Crimes', 'Other Offences', 'Total Development']

# Function to create sequences for RNN training
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x, seq_y = data[i:i+n_steps], data[i+n_steps]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to forecast future values up to a specific year
def forecast_to_2030(model, initial_data, scaler, feature_columns, n_steps, final_year=2030):
    current_data = initial_data[-n_steps:].reshape(1, n_steps, -1)
    predictions = []
    last_known_year = summarized_df['Year'].max()
    years_to_predict = final_year - last_known_year

    for _ in range(years_to_predict):
        next_step = model.predict(current_data)
        predictions.append(next_step)
        current_data = np.append(current_data[:, 1:, :], [next_step], axis=1)

    predictions = np.array(predictions).reshape(years_to_predict, -1)
    predictions_rescaled = scaler.inverse_transform(predictions)
    predictions_df = pd.DataFrame(predictions_rescaled, columns=feature_columns,
                                  index=pd.date_range(start=f'{last_known_year + 1}', periods=years_to_predict, freq='A'))
    return predictions_df

# Normalize the features and prepare data per area
all_predictions = pd.DataFrame()
all_mape = []  # List to store MAPE for each area

for area in summarized_df['Area'].unique():
    area_data = summarized_df[summarized_df['Area'] == area]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(area_data[feature_columns])

    # Define sequence length and create sequences
    n_steps = 3
    X, y = create_sequences(data_scaled, n_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and configure the RNN model using SimpleRNN
    model = Sequential([
        SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(n_steps, X_train.shape[2])),
        Dropout(0.2),
        SimpleRNN(50, activation='relu'),
        Dropout(0.2),
        Dense(len(feature_columns))  # Output layer for multiple features
    ])

    # Compile and train the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

    # Calculate MAPE on the test set
    y_pred_test = model.predict(X_test)
    y_test_inverse = scaler.inverse_transform(y_test)
    y_pred_test_inverse = scaler.inverse_transform(y_pred_test)
    mape = np.mean(np.abs((y_test_inverse - y_pred_test_inverse) / y_test_inverse)) * 100
    all_mape.append(mape)  # Store MAPE for each area

    # Forecast up to 2030 using the trained model
    last_steps_data = data_scaled[-n_steps:]
    predictions_df = forecast_to_2030(model, last_steps_data, scaler, feature_columns, n_steps)
    predictions_df['Area'] = area
    all_predictions = pd.concat([all_predictions, predictions_df], axis=0)

# Calculate the overall MAPE
overall_mape = np.mean(all_mape)
print(f"Overall MAPE across all areas: {overall_mape:.2f}%")