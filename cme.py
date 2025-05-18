import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from io import StringIO, BytesIO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Create the cmev directory if it doesn't exist
os.makedirs("cmev", exist_ok=True)

# Updated NOAA API URL for CME data
CME_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"

def save_to_local(df, filename):
    """Saves DataFrame to local storage"""
    if df.empty:
        print(f"No valid data to save for {filename}.")
        return

    filepath = os.path.join("cmev", filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filename} to local folder cmev")

def save_figure_to_local(fig, filename):
    """Saves matplotlib figure to local storage"""
    filepath = os.path.join("cmev", filename)
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization {filename} to local folder cmev")

def fetch_cme_data():
    """Fetches and stores the latest Coronal Mass Ejection (CME) data."""
    try:
        response = requests.get(CME_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        cme_list = []
        for cme in data:
            cme_info = {
                'Time': cme.get('startTime', ''),
                'Analysis_Time': cme.get('analysisTime', ''),
                'Source_Location': cme.get('sourceLocation', ''),
                'Active_Region': cme.get('activeRegionNum', ''),
                'Type': cme.get('type', ''),
                'Speed_km_s': cme.get('speed', ''),
                'Half_Angle_deg': cme.get('halfAngle', ''),
                'Latitude_deg': cme.get('latitude', ''),
                'Longitude_deg': cme.get('longitude', '')
            }
            cme_list.append(cme_info)

        df = pd.DataFrame(cme_list)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

        # Convert numeric columns to float
        numeric_cols = ['Speed_km_s', 'Half_Angle_deg', 'Latitude_deg', 'Longitude_deg']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by time
        df = df.sort_values('Time')

        # Save raw data
        save_to_local(df, "cme_data.csv")

        # Create and save visualizations
        create_visualizations(df)

        # Train LSTM model and make predictions
        if len(df) > 30:  # Need sufficient data for training
            prediction_df = predict_cme_activity(df)
            save_to_local(prediction_df, "cme_predictions.csv")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch CME data: {e}")
        return pd.DataFrame()

def create_visualizations(df):
    """Create NOAA-style visualizations for CME data"""
    if df.empty:
        print("No data available for visualization.")
        return

    # 1. CME Speed Visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Time'], df['Speed_km_s'], c=df['Speed_km_s'],
                cmap='plasma', alpha=0.8, s=100, edgecolors='black')

    plt.colorbar(label='Speed (km/s)')
    plt.title('Coronal Mass Ejection (CME) Speed Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Speed (km/s)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    save_figure_to_local(plt.gcf(), "cme_speed_visualization.png")
    plt.close()

    # 2. CME Direction Visualization (longitude vs latitude)
    plt.figure(figsize=(10, 10))

    # Create custom colormap from yellow to red
    cmap = LinearSegmentedColormap.from_list('speed_cmap', ['yellow', 'orange', 'red'])

    sc = plt.scatter(df['Longitude_deg'], df['Latitude_deg'],
                     c=df['Speed_km_s'], cmap=cmap,
                     s=df['Half_Angle_deg']*5, alpha=0.7, edgecolors='black')

    plt.colorbar(sc, label='Speed (km/s)')
    plt.title('CME Direction and Angular Width', fontsize=16)
    plt.xlabel('Longitude (degrees)', fontsize=12)
    plt.ylabel('Latitude (degrees)', fontsize=12)
    plt.grid(alpha=0.3)

    # Add Earth at center (0,0)
    plt.scatter([0], [0], c='blue', s=300, marker='o', edgecolors='black', label='Earth')

    # Add circle representing the Sun's limb (90 degrees in each direction)
    circle = plt.Circle((0, 0), 90, fill=False, linestyle='--', color='gray')
    plt.gca().add_artist(circle)

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.legend()

    plt.tight_layout()
    save_figure_to_local(plt.gcf(), "cme_direction_visualization.png")
    plt.close()

    # 3. CME Frequency Histogram by Month
    if not df.empty and 'Time' in df.columns:
        df['Month'] = df['Time'].dt.strftime('%Y-%m')
        monthly_counts = df.groupby('Month').size()

        plt.figure(figsize=(14, 6))
        monthly_counts.plot(kind='bar', color='orange')
        plt.title('Monthly CME Frequency', fontsize=16)
        plt.xlabel('Year-Month', fontsize=12)
        plt.ylabel('Number of CMEs', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        save_figure_to_local(plt.gcf(), "cme_monthly_frequency.png")
        plt.close()

def create_sequences(data, n_steps):
    """Create sequences for LSTM model"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), :])
        y.append(data[i + n_steps, :])
    return np.array(X), np.array(y)

def predict_cme_activity(df):
    """Train LSTM model and predict CME activity for next 3 days"""
    # Filter relevant columns for prediction
    try:
        # Use only speed for prediction as it's the most important parameter
        df_model = df[['Time', 'Speed_km_s']].dropna()

        if len(df_model) < 30:
            print("Not enough data for prediction")
            return pd.DataFrame()

        # Sort by time
        df_model = df_model.sort_values('Time')

        # Resample to daily frequency (taking average of speed)
        df_daily = df_model.set_index('Time').resample('D').mean().fillna(method='ffill')

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_daily[['Speed_km_s']])

        # Create sequences for LSTM
        n_steps = 7  # Use 7 days of history to predict the next day
        X, y = create_sequences(scaled_data, n_steps)

        # Split into train and test sets (using 80% for training)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                  verbose=0, shuffle=False)

        # Make predictions for the next 3 days
        last_sequence = scaled_data[-n_steps:].reshape(1, n_steps, 1)

        predictions = []
        current_sequence = last_sequence.copy()

        # Predict next 3 days
        for _ in range(3):
            next_day_prediction = model.predict(current_sequence, verbose=0)
            predictions.append(next_day_prediction[0, 0])

            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[:, 1:, :],
                                         next_day_prediction.reshape(1, 1, 1),
                                         axis=1)

        # Inverse transform predictions to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        # Create prediction dataframe
        last_date = df_daily.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(3)]

        prediction_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Speed_km_s': predictions.flatten()
        })

        # Create visualization with historical data and predictions
        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(df_daily.index, df_daily['Speed_km_s'], label='Historical Data', color='blue')

        # Plot predictions
        plt.plot(prediction_df['Date'], prediction_df['Predicted_Speed_km_s'],
                 label='3-Day Prediction', color='red', linestyle='--', marker='o')

        plt.title('CME Speed: Historical Data and 3-Day LSTM Prediction', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Speed (km/s)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        save_figure_to_local(plt.gcf(), "cme_prediction_visualization.png")
        plt.close()

        return prediction_df

    except Exception as e:
        print(f"Error in prediction model: {e}")
        return pd.DataFrame()

def check_for_earth_directed_cmes(df):
    """Identify potentially Earth-directed CMEs and assess their impact"""
    if df.empty:
        return

    # Consider CMEs with longitude between -30 and 30 degrees as potentially Earth-directed
    earth_directed = df[(df['Longitude_deg'] >= -30) & (df['Longitude_deg'] <= 30)]

    if not earth_directed.empty:
        # Create a risk assessment visualization
        plt.figure(figsize=(10, 8))

        # Plot CMEs with color based on speed and size based on angular width
        sc = plt.scatter(earth_directed['Longitude_deg'], earth_directed['Latitude_deg'],
                         c=earth_directed['Speed_km_s'], cmap='YlOrRd',
                         s=earth_directed['Half_Angle_deg']*8, alpha=0.7, edgecolors='black')

        plt.colorbar(sc, label='Speed (km/s)')
        plt.title('Potentially Earth-Directed CMEs', fontsize=16)
        plt.xlabel('Longitude (degrees)', fontsize=12)
        plt.ylabel('Latitude (degrees)', fontsize=12)
        plt.grid(alpha=0.3)

        # Add Earth at center (0,0)
        plt.scatter([0], [0], c='blue', s=300, marker='o', edgecolors='black', label='Earth')

        # Add markers for danger zones
        plt.axvspan(-15, 15, alpha=0.2, color='red', label='High Risk Zone')
        plt.axvspan(-30, -15, alpha=0.1, color='orange', label='Medium Risk Zone')
        plt.axvspan(15, 30, alpha=0.1, color='orange')

        plt.xlim(-45, 45)
        plt.ylim(-45, 45)
        plt.legend()

        plt.tight_layout()
        save_figure_to_local(plt.gcf(), "earth_directed_cmes.png")
        plt.close()

        # Create a risk assessment table
        risk_df = earth_directed.copy()

        # Calculate estimated arrival time (rough estimate)
        risk_df['Est_Arrival_Days'] = 150000 / risk_df['Speed_km_s']  # ~150 million km / speed
        risk_df['Est_Arrival_Time'] = risk_df['Time'] + pd.to_timedelta(risk_df['Est_Arrival_Days'], unit='d')

        # Calculate risk level based on speed and direction
        def calculate_risk(row):
            if abs(row['Longitude_deg']) <= 15:
                if row['Speed_km_s'] > 1000:
                    return 'High'
                else:
                    return 'Medium'
            else:
                if row['Speed_km_s'] > 1200:
                    return 'Medium'
                else:
                    return 'Low'

        risk_df['Risk_Level'] = risk_df.apply(calculate_risk, axis=1)

        # Keep only the relevant columns for the risk assessment
        risk_assessment = risk_df[['Time', 'Speed_km_s', 'Longitude_deg', 'Latitude_deg',
                                  'Est_Arrival_Time', 'Risk_Level']].sort_values('Time', ascending=False)

        save_to_local(risk_assessment, "cme_risk_assessment.csv")

# Periodic task scheduling function
def run_cme_monitoring():
    while True:
        print(f"Fetching real-time Coronal Mass Ejection (CME) data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        df = fetch_cme_data()

        if not df.empty:
            check_for_earth_directed_cmes(df)
            print(f"Processed {len(df)} CME events. Visualizations and predictions saved.")

        print("Waiting 10 minutes before next update...")
        time.sleep(600)  # Fetch data every 10 minutes

# Main execution
if __name__ == "__main__":
    run_cme_monitoring()