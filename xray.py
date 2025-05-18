#Xray flux
import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Local storage configuration
LOCAL_STORAGE_DIR = "xray"
OUTPUT_FILENAME = "solar_xray_data.csv"  # Single file for all data

# Create local storage directory if it doesn't exist
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

# NOAA API URL for Solar X-ray data
SOLAR_XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


def save_to_local(df, filename=OUTPUT_FILENAME):
    """
    Saves DataFrame to local storage.
    """
    if df.empty:
        print(f"No valid data to save for {filename}.")
        return

    filepath = os.path.join(LOCAL_STORAGE_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filename} to local storage at {filepath}")


def save_visualization_to_local(fig, filename):
    """
    Saves a matplotlib figure to local storage.
    """
    filepath = os.path.join(LOCAL_STORAGE_DIR, filename)
    fig.savefig(filepath, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization {filename} to local storage at {filepath}")


def fetch_solar_xray_data():
    """
    Fetches solar X-ray flux data from NOAA and handles missing values.
    """
    try:
        response = requests.get(SOLAR_XRAY_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Create DataFrame from the JSON data
        df = pd.DataFrame(data)

        # Ensure required columns exist
        if 'time_tag' not in df.columns or 'flux' not in df.columns:
            print("Error: Required columns missing in data. Check API response.")
            return None

        # Print column info for debugging
        print(f"Columns in raw data: {df.columns.tolist()}")
        print(f"Sample data - first row: {df.iloc[0].to_dict()}")

        # Convert timestamps to datetime format
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce")

        # Handle energy bands - extract as a separate column but keep only numeric flux values
        if 'energy' in df.columns:
            # Store energy band info in a separate column
            df['energy_band'] = df['energy']

        # Convert flux values to numeric - this is essential
        df["flux"] = pd.to_numeric(df["flux"], errors="coerce")

        # Filter out non-numeric rows
        non_numeric_rows = df["flux"].isna().sum()
        if non_numeric_rows > 0:
            print(f"Found {non_numeric_rows} rows with non-numeric flux values. Removing them.")
            df = df.dropna(subset=["flux"])

        # Select only necessary columns for modeling
        model_columns = ['time_tag', 'flux']
        if 'satellite' in df.columns:
            model_columns.append('satellite')
        if 'energy_band' in df.columns:
            model_columns.append('energy_band')

        # Create a clean dataframe for modeling
        model_df = df[model_columns].copy()

        # Handle missing values properly
        model_df = model_df.sort_values('time_tag')
        model_df['flux'] = model_df['flux'].interpolate(method="linear")
        model_df['flux'] = model_df['flux'].ffill()
        model_df['flux'] = model_df['flux'].bfill()

        return model_df

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch solar X-ray data: {e}")
        return None


def preprocess_data(df, sequence_length=48, forecast_steps=72):
    """
    Prepares X-ray flux data for LSTM model training.
    """
    print("Starting preprocessing with data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Data types:", df.dtypes)

    # Create a copy with time for later reference
    df_with_time = df.copy()

    # Set time as index
    df = df.set_index("time_tag").sort_index()

    # Drop any non-numeric columns except the index
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("Error: No numeric columns available for modeling.")
        return None, None, None, None, None, None

    df = df[numeric_cols]

    print(f"Using these numeric columns for modeling: {numeric_cols}")
    print(f"Data shape after filtering numeric columns: {df.shape}")

    # Ensure flux column exists
    if "flux" not in df.columns:
        print("Error: 'flux' column not found in numeric columns")
        return None, None, None, None, None, None

    # Create a separate scaler for flux column only
    flux_scaler = MinMaxScaler()
    flux_values = df[["flux"]].values
    flux_scaler.fit(flux_values)

    # Scaler for the entire dataset
    full_scaler = MinMaxScaler()
    scaled_data = full_scaler.fit_transform(df)
    flux_index = df.columns.get_loc("flux")

    print(f"Scaled data shape: {scaled_data.shape}")
    print(f"Flux index: {flux_index}")

    X, y = [], []

    for i in range(len(scaled_data) - sequence_length - forecast_steps):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length:i + sequence_length + forecast_steps, flux_index])

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        print("Error: Not enough data for training. Check dataset.")
        return None, None, None, None, None, None

    print(f"Final training data shapes: X={X.shape}, y={y.shape}")

    return X, y, flux_scaler, flux_index, df, df_with_time


def build_lstm_model(input_shape, forecast_steps):
    """
    Builds an LSTM model for X-ray flux time series forecasting.
    """
    model = Sequential([
        LSTM(128, activation="relu", return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(forecast_steps)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def generate_future_dates(last_date, num_days=3, hourly=True):
    """
    Generate future date timestamps for forecasting.
    """
    future_dates = []
    hours_to_add = 24 * num_days if hourly else num_days

    for i in range(1, hours_to_add + 1):
        if hourly:
            future_dates.append(last_date + timedelta(hours=i))
        else:
            future_dates.append(last_date + timedelta(days=i))

    return future_dates


def create_noaa_style_visualization(combined_df, timestamp, short_wavelength_df=None):
    """
    Creates NOAA-style visualization of solar X-ray flux data.
    """
    # Create figure with specific dimensions similar to NOAA
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')

    # Set background color to black like NOAA charts
    ax.set_facecolor('black')

    # Split data by type
    historical = combined_df[combined_df['data_type'] == 'historical']
    forecast = combined_df[combined_df['data_type'] == 'prediction']

    # Plot historical data for long wavelength (1-8 Å) - typically shown in red on NOAA charts
    ax.plot(historical['time_tag'], historical['flux'], color='#ff0000', linewidth=1.5,
            label='GOES Long (1-8 Å)')

    # Plot forecast data with different line style
    ax.plot(forecast['time_tag'], forecast['flux'], color='#ff9900', linewidth=1.5,
            linestyle='--', label='Long (1-8 Å) Forecast')

    # If short wavelength data is available, plot it too (0.5-4 Å) - typically in blue on NOAA charts
    if short_wavelength_df is not None:
        short_historical = short_wavelength_df[short_wavelength_df['data_type'] == 'historical']
        short_forecast = short_wavelength_df[short_wavelength_df['data_type'] == 'prediction']

        ax.plot(short_historical['time_tag'], short_historical['flux'], color='#0066ff',
                linewidth=1.5, label='GOES Short (0.5-4 Å)')
        ax.plot(short_forecast['time_tag'], short_forecast['flux'], color='#00ccff',
                linewidth=1.5, linestyle='--', label='Short (0.5-4 Å) Forecast')

    # Add vertical line at the transition point
    if not historical.empty and not forecast.empty:
        transition_time = forecast['time_tag'].min()
        ax.axvline(x=transition_time, color='white', linestyle='--', alpha=0.7)
        ax.text(transition_time, ax.get_ylim()[1] * 0.95, 'Now', rotation=90, ha='right',
                color='white', fontsize=10)

    # Set y-axis to log scale (typical for NOAA X-ray plots)
    ax.set_yscale('log')

    # Add horizontal lines for flare classifications
    flare_levels = {
        'A': 1e-8,
        'B': 1e-7,
        'C': 1e-6,
        'M': 1e-5,
        'X': 1e-4
    }

    for flare_class, level in flare_levels.items():
        ax.axhline(y=level, color='#555555', linestyle='-', alpha=0.5)
        ax.text(ax.get_xlim()[0], level*1.1, flare_class, color='white', fontsize=12, fontweight='bold')

    # Set y-axis limits to match NOAA range (typically A to X)
    ax.set_ylim(1e-9, 1e-3)

    # Add grid but make it subtle
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555555', alpha=0.5)

    # Format the x-axis to show dates clearly
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d\n%H:%M'))
    plt.xticks(rotation=0)

    # Add title and labels
    ax.set_title('GOES X-ray Flux (1-minute data)\nLast 3 Days + 3-Day Forecast',
                 color='white', fontsize=16)
    ax.set_xlabel('Universal Time', color='white', fontsize=14)
    ax.set_ylabel('Watts per square meter', color='white', fontsize=14)

    # Add legend
    legend = ax.legend(loc='upper left', frameon=True)
    legend.get_frame().set_facecolor('#333333')
    for text in legend.get_texts():
        text.set_color('white')

    # Add additional info text as seen in NOAA charts
    plt.figtext(0.01, 0.01, 'Updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
                color='white', fontsize=10)
    plt.figtext(0.99, 0.01, 'AI-Generated Forecast', color='white',
                fontsize=10, ha='right')

    # Change tick colors to white
    ax.tick_params(axis='both', colors='white')

    # Add flare probability scale on the right y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Flare Probability (%)', color='white', fontsize=14)
    ax2.tick_params(axis='y', colors='white')
    ax2.set_ylim(0, 100)

    # Return the figure and filename
    filename = f"noaa_style_xray_flux_{timestamp}.png"
    return fig, filename


def visualize_data(combined_df, test_predictions=None, test_actuals=None, test_times=None):
    """
    Create visualizations of the solar X-ray flux data and predictions.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualization_files = []

    # Set the plotting style
    plt.style.use('ggplot')
    sns.set_context("talk")

    # 1. Create and save NOAA-style visualization
    noaa_fig, noaa_filename = create_noaa_style_visualization(combined_df, timestamp)
    save_visualization_to_local(noaa_fig, noaa_filename)
    visualization_files.append(noaa_filename)
    plt.close(noaa_fig)

    # 2. Historical and Forecast Combined Plot (alternative style)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data
    historical = combined_df[combined_df['data_type'] == 'historical']
    forecast = combined_df[combined_df['data_type'] == 'prediction']

    ax.plot(historical['time_tag'], historical['flux'], 'b-', label='Historical Data', linewidth=2)
    ax.plot(forecast['time_tag'], forecast['flux'], 'r--', label='LSTM Forecast', linewidth=2)

    # Add vertical line at the transition point
    if not historical.empty and not forecast.empty:
        transition_time = forecast['time_tag'].min()
        ax.axvline(x=transition_time, color='gray', linestyle='--', alpha=0.7)
        ax.text(transition_time, ax.get_ylim()[1] * 0.95, 'Now', rotation=90, ha='right')

    # Format the plot
    ax.set_yscale('log')  # Solar flux is often displayed on logarithmic scale
    ax.set_title('Solar X-ray Flux: Historical and 3-Day Forecast', fontsize=16)
    ax.set_xlabel('Date (UTC)', fontsize=14)
    ax.set_ylabel('X-ray Flux (W/m²)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot locally
    filename = f"solar_xray_forecast_{timestamp}.png"
    save_visualization_to_local(fig, filename)
    visualization_files.append(filename)
    plt.close(fig)

    # 3. Model Accuracy Plot (if test data is available)
    if test_predictions is not None and test_actuals is not None and test_times is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        ax2.plot(test_times, test_actuals, 'b-', label='Actual Values', alpha=0.7)
        ax2.plot(test_times, test_predictions, 'g-', label='Predicted Values', alpha=0.7)

        # Calculate and display RMSE
        rmse = np.sqrt(np.mean((test_actuals - test_predictions) ** 2))
        ax2.text(0.05, 0.95, f'RMSE: {rmse:.6f}', transform=ax2.transAxes,
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        ax2.set_yscale('log')
        ax2.set_title('LSTM Model Test Accuracy for Solar X-ray Flux', fontsize=16)
        ax2.set_xlabel('Date (UTC)', fontsize=14)
        ax2.set_ylabel('X-ray Flux (W/m²)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the model accuracy plot locally
        filename2 = f"solar_xray_model_accuracy_{timestamp}.png"
        save_visualization_to_local(fig2, filename2)
        visualization_files.append(filename2)
        plt.close(fig2)

    # 4. Flux Distribution Histogram
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Plot histogram for recent historical data
    ax3.hist(historical['flux'], bins=30, alpha=0.7, color='blue', label='Historical')

    # Add histogram for forecast data with a different color
    ax3.hist(forecast['flux'], bins=30, alpha=0.5, color='red', label='Forecast')

    ax3.set_title('Distribution of Solar X-ray Flux Values', fontsize=16)
    ax3.set_xlabel('X-ray Flux (W/m²)', fontsize=14)
    ax3.set_ylabel('Frequency', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')

    # Use log scale for x-axis to better visualize the distribution
    ax3.set_xscale('log')

    plt.tight_layout()

    # Save the histogram locally
    filename3 = f"solar_xray_distribution_{timestamp}.png"
    save_visualization_to_local(fig3, filename3)
    visualization_files.append(filename3)
    plt.close(fig3)

    # 5. NOAA-style Daily X-ray Flux (last 24 hours)
    if not historical.empty:
        # Filter to last 24 hours
        last_day = historical['time_tag'].max() - timedelta(days=1)
        last_day_data = historical[historical['time_tag'] >= last_day]

        if not last_day_data.empty:
            fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor='black')
            ax4.set_facecolor('black')

            # Plot 24-hour data
            ax4.plot(last_day_data['time_tag'], last_day_data['flux'], color='red', linewidth=1.5)

            # Set y-axis to log scale
            ax4.set_yscale('log')

            # Add flare classification lines
            for flare_class, level in {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}.items():
                ax4.axhline(y=level, color='#555555', linestyle='-', alpha=0.5)
                ax4.text(ax4.get_xlim()[0], level*1.1, flare_class, color='white', fontsize=12, fontweight='bold')

            ax4.set_ylim(1e-9, 1e-3)
            ax4.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555555', alpha=0.5)

            # Format time axis
            ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

            # Add title and labels
            ax4.set_title('GOES X-ray Flux (Last 24 Hours)', color='white', fontsize=16)
            ax4.set_xlabel('Universal Time', color='white', fontsize=14)
            ax4.set_ylabel('Watts per square meter', color='white', fontsize=14)

            # White tick labels
            ax4.tick_params(axis='both', colors='white')

            # Add timestamp
            plt.figtext(0.01, 0.01, 'Updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
                        color='white', fontsize=10)

            # Save locally
            filename4 = f"xray_flux_24h_{timestamp}.png"
            save_visualization_to_local(fig4, filename4)
            visualization_files.append(filename4)
            plt.close(fig4)

    return visualization_files


def train_and_predict(df):
    """
    Trains an LSTM model and predicts the next 3 days of X-ray flux.
    """
    if df is None or df.empty:
        print("No valid data for training.")
        return None, None, None, None, None

    sequence_length = 48  # 48 hours of data to predict from
    forecast_steps = 72  # 3 days ahead (72 hours)

    print("Preprocessing data...")
    X, y, flux_scaler, flux_index, processed_df, df_with_time = preprocess_data(
        df, sequence_length, forecast_steps
    )

    if X is None:
        return None, None, None, None, None

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Split the data for training and validation
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training model with {len(X_train)} sequences...")
    model = build_lstm_model(X_train.shape[1:], forecast_steps=forecast_steps)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    # Save the model locally with the proper extension (.keras or .h5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xray_lstm_model_{timestamp}.keras"
    model_path = os.path.join(LOCAL_STORAGE_DIR, model_filename)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on test data
    print("Evaluating model on test data...")
    test_predictions = model.predict(X_test)

    # Inverse transform to get actual values
    y_test_reshaped = y_test.reshape(-1, 1)
    test_predictions_reshaped = test_predictions.reshape(-1, 1)

    actual_flux = flux_scaler.inverse_transform(y_test_reshaped).reshape(y_test.shape)
    predicted_flux = flux_scaler.inverse_transform(test_predictions_reshaped).reshape(test_predictions.shape)

    # Extract the last hour of each sequence for evaluation
    actual_hourly = actual_flux[:, -1]
    predicted_hourly = predicted_flux[:, -1]

    # Get corresponding time indices for test data
    test_time_indices = processed_df.index[-len(actual_hourly):]

    # Now predict future 3 days beyond the available data
    print("Generating 3-day forecast...")
    last_sequence = processed_df.values[-sequence_length:].copy()

    # Create a scaler specifically for this last sequence
    seq_scaler = MinMaxScaler()
    last_sequence_scaled = seq_scaler.fit_transform(last_sequence)

    # Reshape for prediction
    last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, processed_df.shape[1])

    # Predict the next forecast_steps
    future_scaled = model.predict(last_sequence_scaled)[0]

    # Inverse transform the prediction
    future_scaled_reshaped = future_scaled.reshape(-1, 1)
    future_predictions = flux_scaler.inverse_transform(future_scaled_reshaped).flatten()

    # Generate future dates for the forecast
    last_date = processed_df.index[-1]
    future_dates = generate_future_dates(last_date, num_days=3, hourly=True)

    # Create DataFrame for the forecast
    forecast_df = pd.DataFrame({
        'time_tag': future_dates,
        'flux': future_predictions,
        'data_type': 'prediction'
    })

    return actual_hourly, predicted_hourly, test_time_indices, forecast_df, df_with_time


def main():
    print("Fetching real-time solar X-ray flux data...")
    df = fetch_solar_xray_data()

    if df is not None and not df.empty:
        print(f"Data retrieved successfully. Shape: {df.shape}")

        try:
            # Train model and generate predictions
            actual_hourly, predicted_hourly, test_time_indices, forecast_df, df_with_time = train_and_predict(df)

            if actual_hourly is not None and forecast_df is not None:
                print(
                    f"Prediction successful. Generated {len(actual_hourly)} test predictions and {len(forecast_df)} forecast points")

                # Create combined 3-day historical + 3-day forecast dataset
                # Get last 3 days of historical data
                historical_df = df_with_time.copy()
                historical_df['data_type'] = 'historical'
                historical_df = historical_df.sort_values('time_tag')

                # Calculate timestamp for 3 days ago
                latest_time = historical_df['time_tag'].max()
                three_days_ago = latest_time - timedelta(days=3)

                # Filter for last 3 days only
                recent_historical = historical_df[historical_df['time_tag'] >= three_days_ago]

                # Extract data for different wavelength bands if available
                short_wavelength_df = None
                if 'energy_band' in recent_historical.columns:
                    # Create a copy of short wavelength data (if available)
                    short_mask = recent_historical['energy_band'].str.contains('0.5-4', na=False)
                    if short_mask.any():
                        short_wavelength_df = recent_historical[short_mask].copy()

                        # Generate short wavelength forecast as well
                        # This is simplified - in a production environment, you would train a separate model
                        short_forecast = forecast_df.copy()
                        # Adjust flux values to simulate short wavelength (typically lower than long wavelength)
                        short_forecast['flux'] = short_forecast['flux'] * 0.1

                        # Combine historical and forecast for short wavelength
                        if short_wavelength_df is not None:
                            short_wavelength_df = pd.concat([
                                short_wavelength_df[['time_tag', 'flux', 'data_type']],
                                short_forecast
                            ]).sort_values('time_tag')

                # Select only the necessary columns
                recent_historical = recent_historical[['time_tag', 'flux', 'data_type']]

                # Make sure forecast_df has only the necessary columns
                forecast_df = forecast_df[['time_tag', 'flux', 'data_type']]

                # Combine with forecast data
                combined_df = pd.concat([recent_historical, forecast_df])
                combined_df = combined_df.sort_values('time_tag').reset_index(drop=True)

                # Create visualizations before saving data
                print("Generating visualizations...")
                viz_files = visualize_data(
                    combined_df,
                    test_predictions=predicted_hourly,
                    test_actuals=actual_hourly,
                    test_times=test_time_indices
                )
                print(f"Created visualization files: {viz_files}")

                # Create NOAA-style visualization with both wavelength bands
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                noaa_fig, noaa_filename = create_noaa_style_visualization(
                    combined_df,
                    timestamp,
                    short_wavelength_df=short_wavelength_df
                )
                save_visualization_to_local(noaa_fig, noaa_filename)
                print(f"Created NOAA-style visualization: {noaa_filename}")

                # Save ONLY the combined data - this is our single file
                save_to_local(combined_df)
                print("Data saved successfully to local storage")

        except Exception as e:
            print(f"Error during training or prediction: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Failed to retrieve valid data")


def continuous_monitoring():
    """
    Continuously monitor and fetch solar X-ray data at regular intervals.
    """
    print("Starting continuous solar X-ray flux monitoring...")
    while True:
        try:
            main()
            print("Waiting 10 minutes before next update...")
            time.sleep(600)  # Wait for 10 minutes before next update
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            print("Resuming in 60 seconds...")
            time.sleep(60)  # If there's an error, wait a minute before retrying


if __name__ == "__main__":
    continuous_monitoring()