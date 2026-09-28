import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.title('Demand Forecasting with ARIMA')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('demand_data.csv') # Assuming demand_data.csv is in the same directory
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%Y')
    df.rename(columns={'Month': 'Month-Year'}, inplace=True)
    df.set_index('Month-Year', inplace=True)
    return df

df = load_data()

# Load the trained ARIMA model
@st.cache_resource
def load_model():
    model = joblib.load('arima_model.sav') # Updated to .sav
    return model

model = load_model()

st.subheader('Original Demand Data')
st.write(df)

st.subheader('Forecast Future Demand')

# User input for number of future months to forecast
periods = st.slider('Select number of months to forecast:', 1, 24, 6)

# Make future predictions
future_forecast = model.predict(n_periods=periods)

# Create a DataFrame for the forecast
# Determine the last date in the original data
last_date = df.index.max()
# Generate future dates based on the frequency of the original data (monthly in this case)
future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:] # +1 and [1:] to exclude the last date of original data

forecast_df = pd.DataFrame({'Demand_000L': future_forecast}, index=future_dates)

st.write(f'Forecasting for the next {periods} months:')
st.write(forecast_df)

st.subheader('Demand Forecast Visualization')

# Combine original and forecast data for plotting
combined_df = pd.concat([df['Demand_000L'], forecast_df['Demand_000L']])

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Demand_000L'], label='Historical Demand', color='blue')
ax.plot(forecast_df.index, forecast_df['Demand_000L'], label='Forecasted Demand', color='red', linestyle='--')
ax.set_title('Historical and Forecasted Demand')
ax.set_xlabel('Date')
ax.set_ylabel('Demand (000L)')
ax.legend()
ax.grid(True)
st.pyplot(fig)
