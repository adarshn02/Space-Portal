# AI-Driven Space Weather Forecasting Dashboard

This project presents an AI-powered system designed to monitor and forecast space weather events—such as solar flares, coronal mass ejections (CMEs), and geomagnetic storms—with direct applications in the aviation and telecommunications industries.

## 🌌 Overview

With increasing dependency on satellite navigation and communication systems, timely prediction of space weather disturbances has become critical. This dashboard leverages machine learning techniques, real-time space weather data, and immersive visualizations to provide actionable insights for high-risk sectors.

## 🚀 Features

- 🔭 **LSTM and Random Forest models** for time-series forecasting of solar events.
- ☀️ Real-time data ingestion from NOAA and NASA sources.
- 📦 Cloud-based data storage and processing using **Azure Blob Storage**.
- 🛰️ Analysis of solar wind, X-ray flux, interplanetary magnetic field (IMF), and geomagnetic indices.
- 🗺️ Interactive geographic visualizations built with **Folium**.
- 🎮 Immersive 3D simulations of solar activity using **Unreal Engine**.

## 📊 Use Cases

- ✈️ **Aviation**: Predictive alerts for high-frequency radio blackouts, radiation storms, and polar flight path rerouting.
- 📡 **Telecommunications**: Early warnings for satellite disruptions and ionospheric interference.

## 🧠 Tech Stack

- **Languages**: Python
- **Libraries**: TensorFlow, scikit-learn, matplotlib, pandas, folium, cartopy
- **ML Models**: LSTM, Random Forest Regressor
- **Cloud**: Azure Blob Storage
- **Visualization**: Folium, Unreal Engine

## 📂 Folder Structure

space-portal/
├── aurora.py
├── cme.py
├── xray.py
├── flight.py
├── templates/ (HTML visualizations)
├── space_weather_data/
├── images/
└── README.md

## ⚙️ Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
## ⚙️ Installation & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
## 📝 Important Notes
Ensure your Azure Blob Storage credentials and NOAA/NASA API keys are correctly configured in your environment.

Unreal Engine-based visualizations are built and rendered separately from the Python pipeline.

yaml
Copy
Edit
