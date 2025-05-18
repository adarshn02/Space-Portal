import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from io import StringIO, BytesIO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import json

# Create data directory if it doesn't exist
DATA_DIR = 'space_weather_data'
VISUALIZATIONS_DIR = os.path.join(DATA_DIR, 'visualizations')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Try to import the geomagnetic_latitude module, otherwise define a fallback function.
try:
    from geomagnetic_latitude import calculate_mag_lat_lon
except ModuleNotFoundError:
    def calculate_mag_lat_lon(latitude, longitude):
        """
        Fallback function for calculating magnetic latitude and longitude.
        Currently, it returns the input coordinates unchanged.
        You can replace this with an appropriate calculation if needed.
        """
        return latitude, longitude

# NOAA Space Weather API URLs
KP_INDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
OVATION_AURORA_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
DST_INDEX_URL = "https://services.swpc.noaa.gov/json/dst.json"
SOLAR_WIND_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
IMF_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"

def save_locally(df, filename):
    """Saves DataFrame to local CSV file"""
    if df.empty:
        print(f"No valid data to save for {filename}.")
        return

    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filename} to {filepath}")

def save_figure_locally(fig, filename):
    """Saves matplotlib figure to local file"""
    filepath = os.path.join(VISUALIZATIONS_DIR, filename)
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization {filename} to {filepath}")

def load_locally(filename):
    """Loads a DataFrame from local CSV file"""
    try:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File {filename} does not exist in local storage.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return pd.DataFrame()

def fetch_kp_index_data():
    """Fetches the latest Kp index data from NOAA"""
    try:
        response = requests.get(KP_INDEX_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Skip the header row
        kp_data = []
        for entry in data[1:]:
            kp_info = {
                'Time_Tag': entry[0],
                'Kp_Index': float(entry[1])
            }
            kp_data.append(kp_info)

        df = pd.DataFrame(kp_data)
        if not df.empty:
            # Convert time to datetime
            df['Time_Tag'] = pd.to_datetime(df['Time_Tag'])

            # Sort by time
            df = df.sort_values('Time_Tag')

            # Save raw data locally
            save_locally(df, "kp_index_data.csv")

            # Create visualizations
            create_kp_visualizations(df)

            # Try to load historical data to add to the current data for better predictions
            historical_df = load_locally("historical_kp_data.csv")
            if not historical_df.empty:
                historical_df['Time_Tag'] = pd.to_datetime(historical_df['Time_Tag'])
                # Combine, but avoid duplicates
                combined_df = pd.concat([historical_df, df]).drop_duplicates(subset=['Time_Tag'])
                combined_df = combined_df.sort_values('Time_Tag')
                # Save the updated historical data
                save_locally(combined_df, "historical_kp_data.csv")
                # Use the combined data for predictions
                df = combined_df

            # Make predictions
            predictions = predict_kp_index(df)
            if not predictions.empty:
                save_locally(predictions, "kp_predictions.csv")
                # Generate aurora forecast
                generate_aurora_forecast(predictions)

            return df
        else:
            print("No Kp index data available")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch Kp index data: {e}")
        return pd.DataFrame()

def fetch_solar_wind_data():
    """Fetches solar wind plasma and IMF data from NOAA"""
    try:
        # Fetch plasma data (speed, density, temperature)
        plasma_response = requests.get(SOLAR_WIND_URL, timeout=10)
        plasma_response.raise_for_status()
        plasma_data = plasma_response.json()

        # Fetch magnetic field data (Bz component is crucial for aurora)
        imf_response = requests.get(IMF_URL, timeout=10)
        imf_response.raise_for_status()
        imf_data = imf_response.json()

        # Process plasma data
        plasma_df = pd.DataFrame(plasma_data[1:], columns=plasma_data[0])
        plasma_df.rename(columns={
            'time_tag': 'Time_Tag',
            'density': 'Density',
            'speed': 'Speed',
            'temperature': 'Temperature'
        }, inplace=True)

        # Process IMF data
        imf_df = pd.DataFrame(imf_data[1:], columns=imf_data[0])
        imf_df.rename(columns={
            'time_tag': 'Time_Tag',
            'bx_gsm': 'Bx',
            'by_gsm': 'By',
            'bz_gsm': 'Bz',
            'bt': 'Bt'
        }, inplace=True)

        # Convert columns to appropriate data types
        for df in [plasma_df, imf_df]:
            df['Time_Tag'] = pd.to_datetime(df['Time_Tag'])
            for col in df.columns:
                if col != 'Time_Tag':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Merge the datasets on Time_Tag
        solar_wind_df = pd.merge(plasma_df, imf_df, on='Time_Tag', how='outer')

        # Sort by time and handle missing values
        solar_wind_df = solar_wind_df.sort_values('Time_Tag')
        solar_wind_df = solar_wind_df.interpolate(method='linear')

        # Save to local storage
        save_locally(solar_wind_df, "solar_wind_data.csv")

        # Create visualizations
        create_solar_wind_visualizations(solar_wind_df)

        return solar_wind_df

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch solar wind data: {e}")
        return pd.DataFrame()

def fetch_dst_index():
    """Fetches Dst index data from NOAA"""
    try:
        response = requests.get(DST_INDEX_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        dst_data = []
        for entry in data:
            dst_info = {
                'Time_Tag': entry['time_tag'],
                'Dst_Index': float(entry['dst'])
            }
            dst_data.append(dst_info)

        df = pd.DataFrame(dst_data)
        if not df.empty:
            # Convert time to datetime
            df['Time_Tag'] = pd.to_datetime(df['Time_Tag'])

            # Sort by time
            df = df.sort_values('Time_Tag')

            # Save raw data locally
            save_locally(df, "dst_index_data.csv")

            return df
        else:
            print("No Dst index data available")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch Dst index data: {e}")
        return pd.DataFrame()

def fetch_ovation_aurora():
    """Fetches the latest OVATION aurora forecast from NOAA"""
    try:
        response = requests.get(OVATION_AURORA_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract the forecast time
        forecast_time = data.get('Forecast Time', '')
        coordinates = data.get('coordinates', [])
        aurora_data = data.get('aurora', [])

        # Create a DataFrame for the aurora forecast
        aurora_df = pd.DataFrame({
            'longitude': [coord[0] for coord in coordinates],
            'latitude': [coord[1] for coord in coordinates],
            'aurora_probability': aurora_data
        })

        # Add forecast time to the DataFrame
        aurora_df['forecast_time'] = forecast_time

        # Save raw data locally
        save_locally(aurora_df, "ovation_aurora_data.csv")

        # Create aurora map visualization
        create_aurora_map(aurora_df, forecast_time)

        return aurora_df

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch OVATION aurora data: {e}")
        return pd.DataFrame()

def create_kp_visualizations(df):
    """Creates visualizations for Kp index data"""
    if df.empty:
        print("No data available for Kp index visualization.")
        return

    # 1. Kp Index Time Series
    plt.figure(figsize=(14, 8))

    # Plot Kp index
    plt.plot(df['Time_Tag'], df['Kp_Index'], marker='o', linestyle='-', color='blue')

    # Add threshold lines for different geomagnetic storm levels
    plt.axhline(y=5, color='orange', linestyle='--', label='G1 - Minor Storm')
    plt.axhline(y=6, color='red', linestyle='--', label='G2 - Moderate Storm')
    plt.axhline(y=7, color='purple', linestyle='--', label='G3 - Strong Storm')
    plt.axhline(y=8, color='darkred', linestyle='--', label='G4 - Severe Storm')
    plt.axhline(y=9, color='black', linestyle='--', label='G5 - Extreme Storm')

    plt.title('Planetary K-index (Kp) Over Time', fontsize=16)
    plt.xlabel('Date (UTC)', fontsize=12)
    plt.ylabel('Kp Index', fontsize=12)
    plt.ylim(0, 9.5)  # Kp index ranges from 0 to 9
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    save_figure_locally(plt.gcf(), "kp_index_timeseries.png")
    plt.close()

    # 2. Distribution of Kp values over time (histogram)
    plt.figure(figsize=(12, 6))
    plt.hist(df['Kp_Index'], bins=range(0, 11), alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Kp Index Values', fontsize=16)
    plt.xlabel('Kp Index', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(range(0, 10))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure_locally(plt.gcf(), "kp_index_distribution.png")
    plt.close()

    # 3. Kp index heatmap by month/year
    if len(df) > 30:  # Only create if we have enough data
        df_heatmap = df.copy()
        df_heatmap['Year'] = df_heatmap['Time_Tag'].dt.year
        df_heatmap['Month'] = df_heatmap['Time_Tag'].dt.month

        # Group by year and month, and calculate mean Kp index
        monthly_kp = df_heatmap.groupby(['Year', 'Month'])['Kp_Index'].mean().reset_index()

        # Pivot for heatmap format
        heatmap_data = monthly_kp.pivot(index='Year', columns='Month', values='Kp_Index')

        plt.figure(figsize=(14, 8))

        # Define a colormap from green to red
        cmap = LinearSegmentedColormap.from_list('kp_cmap', ['green', 'yellow', 'orange', 'red', 'darkred'])

        plt.imshow(heatmap_data, cmap=cmap, aspect='auto', interpolation='nearest')
        plt.colorbar(label='Average Kp Index')

        plt.title('Monthly Average Kp Index by Year', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Year', fontsize=12)

        # Set x-ticks to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(range(12), month_names)

        # Set y-ticks to years
        years = heatmap_data.index.tolist()
        plt.yticks(range(len(years)), years)

        plt.tight_layout()
        save_figure_locally(plt.gcf(), "kp_index_monthly_heatmap.png")
        plt.close()

def create_solar_wind_visualizations(df):
    """Creates visualizations for solar wind data"""
    if df.empty:
        print("No data available for solar wind visualization.")
        return

    # 1. Solar Wind Speed and IMF Bz Component
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot solar wind speed
    ax1.plot(df['Time_Tag'], df['Speed'], color='blue', label='Solar Wind Speed')
    ax1.set_xlabel('Date (UTC)', fontsize=12)
    ax1.set_ylabel('Solar Wind Speed (km/s)', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for Bz
    ax2 = ax1.twinx()
    ax2.plot(df['Time_Tag'], df['Bz'], color='red', label='IMF Bz Component')
    ax2.set_ylabel('IMF Bz (nT)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a horizontal line at Bz = 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add title and legend
    plt.title('Solar Wind Speed and IMF Bz Component', fontsize=16)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Format date axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    save_figure_locally(plt.gcf(), "solar_wind_speed_bz.png")
    plt.close()

    # 2. Combined Solar Wind Parameters
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot solar wind density
    ax1.plot(df['Time_Tag'], df['Density'], color='green')
    ax1.set_ylabel('Density (n/cm³)', color='green', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_title('Solar Wind Parameters', fontsize=16)
    ax1.grid(True, alpha=0.3)

    # Plot solar wind temperature
    ax2.plot(df['Time_Tag'], df['Temperature'], color='orange')
    ax2.set_ylabel('Temperature (K)', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.grid(True, alpha=0.3)

    # Plot total IMF strength
    ax3.plot(df['Time_Tag'], df['Bt'], color='purple')
    ax3.set_ylabel('IMF Strength (nT)', color='purple', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_xlabel('Date (UTC)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Format date axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    save_figure_locally(plt.gcf(), "solar_wind_parameters.png")
    plt.close()

    # 3. IMF Components
    plt.figure(figsize=(14, 8))

    plt.plot(df['Time_Tag'], df['Bx'], color='red', label='Bx')
    plt.plot(df['Time_Tag'], df['By'], color='green', label='By')
    plt.plot(df['Time_Tag'], df['Bz'], color='blue', label='Bz')
    plt.plot(df['Time_Tag'], df['Bt'], color='black', linestyle='--', label='Total Field (Bt)')

    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    plt.title('Interplanetary Magnetic Field Components', fontsize=16)
    plt.xlabel('Date (UTC)', fontsize=12)
    plt.ylabel('Magnetic Field (nT)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    save_figure_locally(plt.gcf(), "imf_components.png")
    plt.close()

def create_sequences(data, n_steps):
    """Create sequences for LSTM model"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), :])
        y.append(data[i + n_steps, 0])  # Only predict Kp index
    return np.array(X), np.array(y)

def predict_kp_index(df):
    """Predict Kp index using LSTM model"""
    try:
        # Ensure we have enough data for prediction
        if len(df) < 48:
            print("Not enough data for Kp prediction.")
            return pd.DataFrame()

        # Prepare data for modeling
        df_model = df[['Time_Tag', 'Kp_Index']].copy()

        # Fill any missing values
        df_model['Kp_Index'] = df_model['Kp_Index'].fillna(method='ffill').fillna(method='bfill')

        # Get solar wind data if available to enhance predictions
        solar_wind_df = load_locally("solar_wind_data.csv")

        # Feature engineering
        features = ['Kp_Index']

        # Add solar wind features if available
        if not solar_wind_df.empty:
            solar_wind_df['Time_Tag'] = pd.to_datetime(solar_wind_df['Time_Tag'])

            # Resample both dataframes to 3-hour intervals (Kp standard)
            df_model.set_index('Time_Tag', inplace=True)
            df_model = df_model.resample('3H').mean()

            solar_wind_df.set_index('Time_Tag', inplace=True)
            solar_wind_df = solar_wind_df.resample('3H').mean()

            # Merge dataframes
            merged_df = pd.merge(df_model, solar_wind_df, left_index=True, right_index=True, how='left')
            merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

            # Use important solar wind parameters for prediction
            if 'Bz' in merged_df.columns and 'Speed' in merged_df.columns:
                features.extend(['Bz', 'Speed', 'Bt'])

                # Create additional features
                merged_df['Bz_6h_min'] = merged_df['Bz'].rolling(window=2).min()  # 6-hour minimum (2 * 3h intervals)
                merged_df['Speed_gradient'] = merged_df['Speed'].diff() / 3  # Speed change per hour

                features.extend(['Bz_6h_min', 'Speed_gradient'])

                # Fill NaN values created by diff() and rolling()
                merged_df = merged_df.fillna(method='bfill')

                df_model = merged_df
            else:
                df_model = df_model.reset_index()
        else:
            # If no solar wind data, reset index to have Time_Tag as a column
            df_model = df_model.reset_index()

        # Split the data into features and target
        X_data = df_model[features].values

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X_data)

        # Create sequences for LSTM
        n_steps = 8  # Use 24 hours of history (8 * 3-hour Kp readings)

        # Train two different models:
        # 1. LSTM for short-term forecasting
        # 2. Random Forest for multi-day forecasting

        # LSTM model for short-term (24-hour) forecasting
        X, y = create_sequences(X_scaled, n_steps)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build Bidirectional LSTM model
        model = Sequential([
            Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=(n_steps, len(features))),
            Dropout(0.2),
            Bidirectional(LSTM(32, activation='relu')),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Linear activation for regression
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=16,
                  validation_data=(X_test, y_test),
                  verbose=0, callbacks=[early_stop], shuffle=False)

        # Random Forest for multi-day forecasting
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # For RF, we use a simpler feature set (just the latest data points)
        X_flat = X_scaled[-48:]  # Last 48 data points

        # Train RF on all available data
        rf_model.fit(X_flat, df_model['Kp_Index'].values[-48:])

        # Make predictions
        # Start with the last sequence from our data
        last_sequence = X_scaled[-n_steps:].reshape(1, n_steps, len(features))

        # For RF, we prepare a different input
        rf_input = X_scaled[-1:].reshape(1, -1)

        # Prediction lists
        lstm_predictions = []
        rf_predictions = []

        # Predict next 24 hours (8 Kp values, 3 hours apart)
        for i in range(8):
            # LSTM prediction
            next_lstm_pred = model.predict(last_sequence, verbose=0)[0][0]
            lstm_predictions.append(next_lstm_pred)

            # RF prediction
            if i == 0:
                next_rf_pred = rf_model.predict(rf_input)[0]
            else:
                # Shift the input for RF and add the last prediction
                rf_input = np.roll(rf_input, -1, axis=1)
                rf_input[0, -1] = next_rf_pred
                next_rf_pred = rf_model.predict(rf_input)[0]

            rf_predictions.append(next_rf_pred)

            # Update sequence for next LSTM prediction
            # We need to create a new feature vector with all features
            if len(features) > 1:
                # For multiple features, we need to use the latest values from our data
                # and only update the Kp index
                new_feature_vector = np.zeros(len(features))
                new_feature_vector[0] = next_lstm_pred  # Set Kp index
                # Copy the last values for other features
                for j in range(1, len(features)):
                    new_feature_vector[j] = X_scaled[-1, j]
            else:
                new_feature_vector = np.array([next_lstm_pred])

            # Update the sequence by removing the first element and adding prediction
            last_sequence = np.append(last_sequence[:, 1:, :],
                                     new_feature_vector.reshape(1, 1, len(features)),
                                     axis=1)

        # Convert predictions back to original scale
        # Create inverse transformation arrays
        lstm_pred_array = np.zeros((len(lstm_predictions), len(features)))
        lstm_pred_array[:, 0] = lstm_predictions  # Set first column to Kp predictions

        rf_pred_array = np.zeros((len(rf_predictions), len(features)))
        rf_pred_array[:, 0] = rf_predictions

        # Inverse transform
        lstm_pred_original = scaler.inverse_transform(lstm_pred_array)[:, 0]
        rf_pred_original = scaler.inverse_transform(rf_pred_array)[:, 0]

        # Ensure values are in valid Kp range (0-9)
        lstm_pred_original = np.clip(lstm_pred_original, 0, 9)
        rf_pred_original = np.clip(rf_pred_original, 0, 9)

        # Create a weighted ensemble (giving more weight to LSTM for short-term)
        ensemble_weights = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        ensemble_predictions = ensemble_weights * lstm_pred_original + (1 - ensemble_weights) * rf_pred_original

        # Round to nearest NOAA standard Kp format (0, 0.3, 0.7, 1, 1.3, 1.7, etc.)
        def round_to_kp(x):
            base = int(x)
            frac = x - base
            if frac < 0.15:
                return float(base)
            elif frac < 0.5:
                return base + 0.3
            elif frac < 0.85:
                return base + 0.7
            else:
                return float(base + 1)

        # Apply rounding
        ensemble_predictions = np.array([round_to_kp(x) for x in ensemble_predictions])

        # Create prediction dataframe
        last_time = df['Time_Tag'].max()
        prediction_times = [last_time + timedelta(hours=3*(i+1)) for i in range(8)]

        prediction_df = pd.DataFrame({
            'Time_Tag': prediction_times,
            'Predicted_Kp_Index': ensemble_predictions
        })

        # Create visualization with historical data and predictions
        plt.figure(figsize=(14, 8))

        # Plot historical data (last 7 days)
        recent_df = df[df['Time_Tag'] > (df['Time_Tag'].max() - timedelta(days=7))]
        plt.plot(recent_df['Time_Tag'], recent_df['Kp_Index'],
               label='Historical Kp Index', color='blue', marker='o')

        # Plot predictions
        plt.plot(prediction_df['Time_Tag'], prediction_df['Predicted_Kp_Index'],
               label='Predicted Kp Index', color='red', marker='x', linestyle='--')

        # Add threshold lines for different geomagnetic storm levels
        plt.axhline(y=5, color='orange', linestyle='--', label='G1 - Minor Storm')
        plt.axhline(y=6, color='red', linestyle='--', label='G2 - Moderate Storm')
        plt.axhline(y=7, color='purple', linestyle='--', label='G3 - Strong Storm')
        plt.axhline(y=8, color='darkred', linestyle='--', label='G4 - Severe Storm')
        plt.axhline(y=9, color='black', linestyle='--', label='G5 - Extreme Storm')

        plt.title('Kp Index Forecast', fontsize=16)
        plt.xlabel('Date (UTC)', fontsize=12)
        plt.ylabel('Kp Index', fontsize=12)
        plt.ylim(0, 9.5)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()
        save_figure_locally(plt.gcf(), "kp_forecast.png")
        plt.close()

        return prediction_df

    except Exception as e:
        print(f"Error in Kp index prediction: {e}")
        return pd.DataFrame()
def create_aurora_map(aurora_df, forecast_time):
    """Creates a global map of aurora visibility"""
    if aurora_df.empty:
        print("No aurora data available for visualization.")
        return

    try:
        # Create figure with Cartopy projection
        plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=90))

        # Add coastlines and features
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.gridlines()

        # Convert geomagnetic coordinates to geographic coordinates if needed
        # (simplified approach, just using original coordinates in this example)
        lons = aurora_df['longitude'].values
        lats = aurora_df['latitude'].values
        aurora_prob = aurora_df['aurora_probability'].values

        # Create a scatter plot with aurora probability as color
        sc = ax.scatter(lons, lats, c=aurora_prob, s=3, 
                        transform=ccrs.PlateCarree(),
                        cmap='viridis', vmin=0, vmax=100)

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label('Aurora Viewing Probability (%)')

        # Add title with forecast time
        plt.title(f'Aurora Forecast for {forecast_time}', fontsize=16)

        # Add text to show the magnetic midnight
        now = datetime.now()
        magnetic_midnight_lon = ((now.hour + 12) % 24) * 15 - 180
        ax.text(0, -0.12, f'Magnetic Midnight: ~{magnetic_midnight_lon}°E', 
                transform=ax.transAxes, ha='center', fontsize=10)

        plt.tight_layout()
        save_figure_locally(plt.gcf(), "aurora_forecast_north.png")
        plt.close()

        # Create a second map for Southern Hemisphere
        plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))

        # Add coastlines and features
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.gridlines()

        # Create a scatter plot for southern hemisphere
        sc = ax.scatter(lons, lats, c=aurora_prob, s=3, 
                        transform=ccrs.PlateCarree(),
                        cmap='viridis', vmin=0, vmax=100)

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label('Aurora Viewing Probability (%)')

        # Add title with forecast time
        plt.title(f'Aurora Forecast for {forecast_time} (South)', fontsize=16)

        plt.tight_layout()
        save_figure_locally(plt.gcf(), "aurora_forecast_south.png")
        plt.close()

    except Exception as e:
        print(f"Error creating aurora map: {e}")

def generate_aurora_forecast(kp_predictions):
    """Generates a simplified aurora forecast based on Kp predictions"""
    if kp_predictions.empty:
        print("No Kp predictions available for aurora forecast.")
        return

    try:
        # Define latitude thresholds for aurora visibility based on Kp index
        # These are approximations based on the general rule: 
        # Aurora visibility latitude ~= 67.5 - 2.5 * Kp (for northern hemisphere)
        kp_to_latitude_map = {
            0: 67.5,
            1: 65.0,
            2: 62.5,
            3: 60.0,
            4: 57.5,
            5: 55.0,
            6: 52.5,
            7: 50.0,
            8: 47.5,
            9: 45.0
        }

        # Create a grid of latitudes and longitudes
        lat_range = np.linspace(-90, 90, 181)  # 1-degree resolution
        lon_range = np.linspace(-180, 180, 361)  # 1-degree resolution
        
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
        
        # Prediction times
        prediction_times = kp_predictions['Time_Tag'].tolist()
        
        # Generate aurora visibility maps for each prediction time
        for i, (time, kp) in enumerate(zip(kp_predictions['Time_Tag'], kp_predictions['Predicted_Kp_Index'])):
            # Get the latitude threshold for this Kp index
            # Round Kp to nearest integer for mapping
            kp_rounded = round(kp)
            if kp_rounded > 9:
                kp_rounded = 9
            elif kp_rounded < 0:
                kp_rounded = 0
                
            threshold_lat = kp_to_latitude_map[kp_rounded]
            
            # Generate aurora probability
            aurora_prob = np.zeros_like(lat_grid)
            
            # Northern hemisphere aurora oval
            distance_from_north_oval = np.abs(lat_grid - (90 - threshold_lat))
            aurora_prob[lat_grid > 0] = np.exp(-0.1 * distance_from_north_oval[lat_grid > 0]**2) * 100
            
            # Southern hemisphere aurora oval (mirror of northern)
            distance_from_south_oval = np.abs(lat_grid - (-90 + threshold_lat))
            aurora_prob[lat_grid < 0] = np.exp(-0.1 * distance_from_south_oval[lat_grid < 0]**2) * 100
            
            # Flatten the arrays for storage
            aurora_data = {
                'longitude': lon_grid.flatten(),
                'latitude': lat_grid.flatten(),
                'aurora_probability': aurora_prob.flatten(),
                'forecast_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Create DataFrame
            aurora_df = pd.DataFrame(aurora_data)
            
            # Save to file
            forecast_filename = f"aurora_forecast_{i+1}.csv"
            save_locally(aurora_df, forecast_filename)
            
            # Create visualization
            if i == 0:  # Only visualize the first forecast (3 hours ahead)
                create_aurora_map(aurora_df, time.strftime('%Y-%m-%d %H:%M:%S'))
                
        # Generate summary of aurora visibility
        cities = {
            'Anchorage': (61.2, -149.9),
            'Reykjavik': (64.1, -21.9),
            'Oslo': (59.9, 10.7),
            'Stockholm': (59.3, 18.1),
            'Helsinki': (60.2, 24.9),
            'Fairbanks': (64.8, -147.7),
            'Edmonton': (53.5, -113.5),
            'Tromsø': (69.6, 18.9),
            'Yellowknife': (62.5, -114.4),
            'Rovaniemi': (66.5, 25.7),
            'Inverness': (57.5, -4.2),
            'Murmansk': (69.0, 33.1),
            'Nuuk': (64.2, -51.7),
            'Dunedin': (-45.9, 170.5),
            'Hobart': (-42.9, 147.3),
            'Ushuaia': (-54.8, -68.3),
            'Punta Arenas': (-53.2, -70.9)
        }
        
        # Calculate visibility probability for each city for the next 24 hours
        city_visibility = []
        
        for city, (lat, lon) in cities.items():
            # For each Kp prediction, calculate if aurora might be visible
            for i, (time, kp) in enumerate(zip(kp_predictions['Time_Tag'], kp_predictions['Predicted_Kp_Index'])):
                kp_rounded = round(kp)
                if kp_rounded > 9:
                    kp_rounded = 9
                elif kp_rounded < 0:
                    kp_rounded = 0
                    
                threshold_lat = kp_to_latitude_map[kp_rounded]
                
                # Check if city is within viewing latitude
                visible = False
                probability = 0
                
                # Northern hemisphere
                if lat > 0 and lat >= (90 - threshold_lat - 5):
                    # Calculate probability based on distance from optimal viewing latitude
                    distance = abs(lat - (90 - threshold_lat))
                    probability = max(0, 100 - distance * 20)  # Decrease by 20% per degree away
                    visible = probability > 10
                # Southern hemisphere
                elif lat < 0 and abs(lat) >= (90 - threshold_lat - 5):
                    # Calculate probability based on distance from optimal viewing latitude
                    distance = abs(abs(lat) - (90 - threshold_lat))
                    probability = max(0, 100 - distance * 20)
                    visible = probability > 10
                
                city_visibility.append({
                    'City': city,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Forecast_Time': time,
                    'Kp_Index': kp,
                    'Visibility_Probability': probability,
                    'Visible': visible
                })
        
        # Create DataFrame for city visibility
        city_visibility_df = pd.DataFrame(city_visibility)
        save_locally(city_visibility_df, "aurora_city_visibility.csv")
        
        return city_visibility_df
            
    except Exception as e:
        print(f"Error generating aurora forecast: {e}")
        return pd.DataFrame()

def calculate_solar_activity_index():
    """Calculate a custom index of overall solar activity"""
    try:
        # Load data
        kp_df = load_locally("kp_index_data.csv")
        dst_df = load_locally("dst_index_data.csv")
        solar_wind_df = load_locally("solar_wind_data.csv")
        
        if kp_df.empty and dst_df.empty and solar_wind_df.empty:
            print("No data available for solar activity index calculation.")
            return pd.DataFrame()
        
        # Process Kp index data
        if not kp_df.empty:
            kp_df['Time_Tag'] = pd.to_datetime(kp_df['Time_Tag'])
            kp_df.set_index('Time_Tag', inplace=True)
            
        # Process Dst index data
        if not dst_df.empty:
            dst_df['Time_Tag'] = pd.to_datetime(dst_df['Time_Tag'])
            dst_df.set_index('Time_Tag', inplace=True)
        
        # Process solar wind data
        if not solar_wind_df.empty:
            solar_wind_df['Time_Tag'] = pd.to_datetime(solar_wind_df['Time_Tag'])
            solar_wind_df.set_index('Time_Tag', inplace=True)
        
        # Create a common timeline with 1-hour resolution
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        timeline = pd.date_range(start=start_time, end=end_time, freq='1H')
        
        # Create an empty DataFrame with the timeline as index
        activity_index = pd.DataFrame(index=timeline)
        
        # Resample and interpolate each dataset to the common timeline
        if not kp_df.empty:
            kp_resampled = kp_df.resample('1H').mean().reindex(timeline)
            kp_resampled.interpolate(method='time', inplace=True)
            activity_index['Kp_Index'] = kp_resampled['Kp_Index']
        
        if not dst_df.empty:
            dst_resampled = dst_df.resample('1H').mean().reindex(timeline)
            dst_resampled.interpolate(method='time', inplace=True)
            activity_index['Dst_Index'] = dst_resampled['Dst_Index']
        
        if not solar_wind_df.empty:
            # Extract key solar wind parameters
            sw_params = ['Speed', 'Bz', 'Bt']
            for param in sw_params:
                if param in solar_wind_df.columns:
                    sw_resampled = solar_wind_df[param].resample('1H').mean().reindex(timeline)
                    sw_resampled.interpolate(method='time', inplace=True)
                    activity_index[param] = sw_resampled
        
        # Fill any remaining NaN values
        activity_index.fillna(method='ffill', inplace=True)
        activity_index.fillna(method='bfill', inplace=True)
        
        # Calculate the solar activity index
        # Normalize each component
        normalized_components = pd.DataFrame(index=activity_index.index)
        
        if 'Kp_Index' in activity_index.columns:
            normalized_components['Kp_norm'] = activity_index['Kp_Index'] / 9.0  # Kp max is 9
        
        if 'Dst_Index' in activity_index.columns:
            # Dst is negative during storms, normalize to 0-1 with 1 being most active
            dst_min = -500  # Extreme storm value
            normalized_components['Dst_norm'] = (activity_index['Dst_Index'] - 0) / (dst_min - 0)
            normalized_components['Dst_norm'] = 1 - normalized_components['Dst_norm'].clip(0, 1)
        
        if 'Speed' in activity_index.columns:
            # Solar wind speed, normalize with 300 km/s as min and 800 km/s as max
            normalized_components['Speed_norm'] = (activity_index['Speed'] - 300) / (800 - 300)
            normalized_components['Speed_norm'] = normalized_components['Speed_norm'].clip(0, 1)
        
        if 'Bz' in activity_index.columns:
            # Bz is important when negative, normalize to 0-1 with 1 being most active
            normalized_components['Bz_norm'] = (activity_index['Bz'] - 0) / (-20 - 0)
            normalized_components['Bz_norm'] = 1 - normalized_components['Bz_norm'].clip(0, 1)
        
        if 'Bt' in activity_index.columns:
            # Total IMF strength, normalize with 5 nT as min and 30 nT as max
            normalized_components['Bt_norm'] = (activity_index['Bt'] - 5) / (30 - 5)
            normalized_components['Bt_norm'] = normalized_components['Bt_norm'].clip(0, 1)
        
        # Calculate the overall solar activity index
        # Weights for each component
        weights = {
            'Kp_norm': 0.3,
            'Dst_norm': 0.3,
            'Speed_norm': 0.2,
            'Bz_norm': 0.15,
            'Bt_norm': 0.05
        }
        
        # Only use available components
        available_components = [col for col in weights.keys() if col in normalized_components.columns]
        if not available_components:
            print("No components available for index calculation.")
            return pd.DataFrame()
        
        # Adjust weights based on available components
        total_weight = sum(weights[comp] for comp in available_components)
        adjusted_weights = {comp: weights[comp]/total_weight for comp in available_components}
        
        # Calculate weighted sum
        activity_index['Solar_Activity_Index'] = sum(
            normalized_components[comp] * adjusted_weights[comp] for comp in available_components
        )
        
        # Add activity level categorization
        activity_levels = [
            (0.0, 0.2, 'Quiet'),
            (0.2, 0.4, 'Low'),
            (0.4, 0.6, 'Moderate'),
            (0.6, 0.8, 'Active'),
            (0.8, 1.0, 'Severe')
        ]
        
        def get_activity_level(value):
            for low, high, level in activity_levels:
                if low <= value < high:
                    return level
            return 'Extreme'  # If value is 1.0 or above
        
        activity_index['Activity_Level'] = activity_index['Solar_Activity_Index'].apply(get_activity_level)
        
        # Save the activity index
        activity_index_df = activity_index.reset_index()
        activity_index_df.rename(columns={'index': 'Time_Tag'}, inplace=True)
        save_locally(activity_index_df, "solar_activity_index.csv")
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        plt.plot(activity_index.index, activity_index['Solar_Activity_Index'], color='red', linewidth=2)
        
        # Add colored bands for activity levels
        for low, high, level in activity_levels:
            plt.axhspan(low, high, alpha=0.2, color={
                'Quiet': 'green',
                'Low': 'blue',
                'Moderate': 'yellow',
                'Active': 'orange',
                'Severe': 'red'
            }.get(level, 'purple'))
        
        plt.title('Solar Activity Index', fontsize=16)
        plt.xlabel('Date (UTC)', fontsize=12)
        plt.ylabel('Activity Index (0-1)', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add a legend for activity levels
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color='green', alpha=0.2, label='Quiet (0.0-0.2)'),
            mpatches.Patch(color='blue', alpha=0.2, label='Low (0.2-0.4)'),
            mpatches.Patch(color='yellow', alpha=0.2, label='Moderate (0.4-0.6)'),
            mpatches.Patch(color='orange', alpha=0.2, label='Active (0.6-0.8)'),
            mpatches.Patch(color='red', alpha=0.2, label='Severe (0.8-1.0)'),
            mpatches.Patch(color='purple', alpha=0.2, label='Extreme (>1.0)')
        ]
        plt.legend(handles=handles, loc='upper left')
        
        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        save_figure_locally(plt.gcf(), "solar_activity_index.png")
        plt.close()
        
        return activity_index_df
        
    except Exception as e:
        print(f"Error calculating solar activity index: {e}")
        return pd.DataFrame()

def generate_space_weather_report():
    """Generates a comprehensive space weather report"""
    try:
        # Initialize report data
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'summary': {},
            'kp_index': {},
            'dst_index': {},
            'solar_wind': {},
            'aurora_forecast': {},
            'alerts': []
        }
        
        # Collect latest data
        kp_df = load_locally("kp_index_data.csv")
        dst_df = load_locally("dst_index_data.csv")
        solar_wind_df = load_locally("solar_wind_data.csv")
        kp_predictions = load_locally("kp_predictions.csv")
        city_visibility = load_locally("aurora_city_visibility.csv")
        
        # Process Kp index data
        if not kp_df.empty:
            kp_df['Time_Tag'] = pd.to_datetime(kp_df['Time_Tag'])
            latest_kp = kp_df.loc[kp_df['Time_Tag'].idxmax()]
            
            report['kp_index'] = {
                'latest_value': float(latest_kp['Kp_Index']),
                'timestamp': latest_kp['Time_Tag'].strftime('%Y-%m-%d %H:%M:%S UTC'),
                'storm_level': get_storm_level(latest_kp['Kp_Index'])
            }
            
            # Check for active geomagnetic storm
            if latest_kp['Kp_Index'] >= 5:
                report['alerts'].append({
                    'level': 'warning',
                    'message': f"Geomagnetic storm in progress (Kp={latest_kp['Kp_Index']}, {get_storm_level(latest_kp['Kp_Index'])})"
                })
        
        # Process Dst index data
        if not dst_df.empty:
            dst_df['Time_Tag'] = pd.to_datetime(dst_df['Time_Tag'])
            latest_dst = dst_df.loc[dst_df['Time_Tag'].idxmax()]
            
            report['dst_index'] = {
                'latest_value': float(latest_dst['Dst_Index']),
                'timestamp': latest_dst['Time_Tag'].strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
            # Check for severe Dst values (strong ring current, typical of geomagnetic storms)
            if latest_dst['Dst_Index'] <= -100:
                report['alerts'].append({
                    'level': 'warning',
                    'message': f"Strong ring current detected (Dst={latest_dst['Dst_Index']} nT)"
                })
        
        # Process solar wind data
        if not solar_wind_df.empty:
            solar_wind_df['Time_Tag'] = pd.to_datetime(solar_wind_df['Time_Tag'])
            latest_sw = solar_wind_df.loc[solar_wind_df['Time_Tag'].idxmax()]
            
            sw_data = {}
            for param in ['Speed', 'Density', 'Temperature', 'Bz', 'Bt']:
                if param in latest_sw:
                    sw_data[param.lower()] = float(latest_sw[param])
            
            sw_data['timestamp'] = latest_sw['Time_Tag'].strftime('%Y-%m-%d %H:%M:%S UTC')
            report['solar_wind'] = sw_data
            
            # Check for solar wind anomalies
            if 'speed' in sw_data and sw_data['speed'] > 500:
                report['alerts'].append({
                    'level': 'info',
                    'message': f"High solar wind speed detected ({sw_data['speed']} km/s)"
                })
            
            if 'bz' in sw_data and sw_data['bz'] < -10:
                report['alerts'].append({
                    'level': 'warning',
                    'message': f"Strong southward IMF Bz detected ({sw_data['bz']} nT)"
                })
        
        # Process aurora forecast
        if not city_visibility.empty:
            city_visibility['Forecast_Time'] = pd.to_datetime(city_visibility['Forecast_Time'])
            
            # Group by city and get maximum visibility probability in the next 24 hours
            city_max_prob = city_visibility.groupby('City')['Visibility_Probability'].max().reset_index()
            
            # Sort cities by visibility probability
            city_max_prob = city_max_prob.sort_values('Visibility_Probability', ascending=False)
            
            # Take top 5 cities with highest probability
            top_cities = city_max_prob.head(5).to_dict('records')
            
            report['aurora_forecast'] = {
                'best_locations': top_cities
            }
            
            # Check if any location has high aurora visibility
            if city_max_prob['Visibility_Probability'].max() > 70:
                report['alerts'].append({
                    'level': 'info',
                    'message': f"High probability of aurora at {city_max_prob.iloc[0]['City']} ({city_max_prob.iloc[0]['Visibility_Probability']:.1f}%)"
                })
        
        # Process Kp predictions
        if not kp_predictions.empty:
            kp_predictions['Time_Tag'] = pd.to_datetime(kp_predictions['Time_Tag'])
            
            # Get maximum predicted Kp in the next 24 hours
            max_kp = kp_predictions['Predicted_Kp_Index'].max()
            max_kp_time = kp_predictions.loc[kp_predictions['Predicted_Kp_Index'].idxmax(), 'Time_Tag']
            
            report['summary']['forecast'] = {
                'max_kp_forecast': float(max_kp),
                'max_kp_time': max_kp_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'storm_level_forecast': get_storm_level(max_kp)
            }
            
            # Check for predicted geomagnetic storm
            if max_kp >= 5:
                report['alerts'].append({
                    'level': 'info',
                    'message': f"Geomagnetic storm forecasted (Kp={max_kp:.1f}, {get_storm_level(max_kp)}) at {max_kp_time.strftime('%Y-%m-%d %H:%M')}"
                })
        
        # Generate overall space weather condition summary
        activity_index_df = load_locally("solar_activity_index.csv")
        if not activity_index_df.empty:
            activity_index_df['Time_Tag'] = pd.to_datetime(activity_index_df['Time_Tag'])
            latest_activity = activity_index_df.loc[activity_index_df['Time_Tag'].idxmax()]
            
            report['summary']['current_condition'] = {
                'activity_index': float(latest_activity['Solar_Activity_Index']),
                'activity_level': latest_activity['Activity_Level'],
                'timestamp': latest_activity['Time_Tag'].strftime('%Y-%m-%d %H:%M:%S UTC')
            }
        else:
            # Fallback if activity index is not available
            if 'kp_index' in report:
                kp_val = report['kp_index']['latest_value']
                report['summary']['current_condition'] = {
                    'activity_level': get_storm_level(kp_val)
                }
        
        # Save report as JSON
        report_path = os.path.join(DATA_DIR, "space_weather_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Space weather report generated and saved to {report_path}")
        return report
        
    except Exception as e:
        print(f"Error generating space weather report: {e}")
        return {}

def get_storm_level(kp_index):
    """Converts Kp index to NOAA geomagnetic storm scale"""
    if kp_index < 5:
        return "G0 (Quiet)"
    elif kp_index < 6:
        return "G1 (Minor)"
    elif kp_index < 7:
        return "G2 (Moderate)"
    elif kp_index < 8:
        return "G3 (Strong)"
    elif kp_index < 9:
        return "G4 (Severe)"
    else:
        return "G5 (Extreme)"

def main():
    """Main function to run the space weather monitor"""
    print("Space Weather Monitor starting...")
    
    # Create data directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    try:
        # Fetch current space weather data
        print("Fetching Kp index data...")
        kp_df = fetch_kp_index_data()
        
        print("Fetching solar wind data...")
        solar_wind_df = fetch_solar_wind_data()
        
        print("Fetching Dst index...")
        dst_df = fetch_dst_index()
        
        print("Fetching OVATION aurora forecast...")
        aurora_df = fetch_ovation_aurora()
        
        # Calculate solar activity index
        print("Calculating solar activity index...")
        calculate_solar_activity_index()
        
        # Generate comprehensive report
        print("Generating space weather report...")
        report = generate_space_weather_report()
        
        print("Space Weather Monitor completed successfully.")
        
        # Print summary
        if report and 'summary' in report:
            if 'current_condition' in report['summary']:
                print(f"\nCurrent Space Weather Condition: {report['summary']['current_condition'].get('activity_level', 'Unknown')}")
                
            if 'forecast' in report['summary']:
                print(f"Forecast Max Kp: {report['summary']['forecast'].get('max_kp_forecast', 'Unknown')} " +
                      f"({report['summary']['forecast'].get('storm_level_forecast', 'Unknown')})")
        
        if report and 'alerts' in report and report['alerts']:
            print("\nAlerts:")
            for alert in report['alerts']:
                print(f"- {alert['message']}")
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()