import streamlit as st
import joblib
import pandas as pd

# Load the ARIMA model
@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model

model = load_model('arima_model.sav')

st.title('ARIMA Demand Forecasting App')
st.write('This app forecasts future demand using a pre-trained ARIMA model.')

# User input for number of future periods
periods = st.slider(
    'Select the number of future months to forecast:',
    min_value=1,
    max_value=24,
    value=6
)

if st.button('Generate Forecast'):
    # Make predictions
    forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)

    # Create a date index for the forecast
    # Assuming the last date in your training data was 2024-07-01 based on your notebook state
    # You might need to adjust the start date if your training data ends differently
    last_date = pd.to_datetime('2024-07-01') # Based on the end of your train_data
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')

    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Demand (000L)': forecast,
        'Lower_Bound': conf_int[:, 0],
        'Upper_Bound': conf_int[:, 1]
    })
    forecast_df.set_index('Date', inplace=True)

    st.subheader(f'Forecast for the next {periods} months:')
    st.dataframe(forecast_df)

    st.subheader('Forecast Visualization:')
    st.line_chart(forecast_df[['Forecasted Demand (000L)', 'Lower_Bound', 'Upper_Bound']])

