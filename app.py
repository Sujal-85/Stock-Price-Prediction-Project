import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, GRU, Dense, SimpleRNN, concatenate, Dropout, Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import datetime
import seaborn as sns
from PIL import Image
import base64
import os
import hashlib
import pickle
import time
from keras import backend as K
from pmdarima import auto_arima

# Create directories for saving models and scalers if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 40px !important;
        color: #2a3f5f;
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .model-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: white;
    }
    .stock-header {
        color: #2a3f5f;
        border-bottom: 2px solid #2a3f5f;
        padding-bottom: 10px;
    }
    .linear-reg-card {
        border-left: 5px solid #4b8df8;
    }
    .lstm-card {
        border-left: 5px solid #ff6b6b;
    }
    .gru-card {
        border-left: 5px solid #51cf66;
    }
    .rnn-card {
        border-left: 5px solid #fcc419;
    }
    .future-pred {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .metric-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .arima-card {
        border-left: 5px solid #845ef7;
    }
    .prophet-card {
        border-left: 5px solid #20c997;
    }
    .hybrid-card {
        border-left: 5px solid #f76707;
    }
    </style>
    """, unsafe_allow_html=True)

# Help section in sidebar
with st.sidebar.expander("ℹ️ Help & Instructions", expanded=False):
    st.markdown("""
        ### How to Use This App
        
        1. **Select a Stock**:
        - Choose from popular stocks or enter a custom symbol
        - For Indian stocks, use `.NS` (NSE) or `.BO` (BSE) suffix
        
        2. **Set Date Range**:
        - Wider ranges provide more training data
        - Future dates will generate forecasts
        
        3. **Choose Models**:
        - Linear Regression: Fast and simple
        - LSTM/GRU: More complex, may need more data
        
        4. **Adjust Parameters**:
        - Epochs: More epochs may improve accuracy (but risk overfitting)
        - Sequence Length: How many past days to consider
        
        5. **Interpret Results**:
        - Compare metrics across models
        - Check prediction vs actual plots
        - View future forecasts if applicable
        """)

# Function to generate a unique model identifier
def generate_model_id(stock, start_date, end_date, model_type, sequence_length, epochs, batch_size):
    """Generate a unique hash ID for a model based on its parameters"""
    params = f"{stock}_{start_date}_{end_date}_{model_type}_{sequence_length}_{epochs}_{batch_size}"
    return hashlib.md5(params.encode()).hexdigest()

# App Header
st.markdown('<h1 class="main-title">📈 Advanced Stock Price Prediction</h1>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Prediction Settings")
    st.expander("", expanded=True)
    
    # Popular stock symbols
    popular_stocks = {
        "Google": "GOOG",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Tesla": "TSLA",
        "Amazon": "AMZN",
        "Facebook": "META",
        "Netflix": "NFLX",
        "Nvidia": "NVDA",
        "Alibaba": "BABA",
        "Twitter": "TWTR"
    }
    
    # Stock selection
    stock_option = st.radio(
        "Stock Selection Method",
        ["Select from popular stocks", "Enter custom stock symbol"],
        index=0
    )
    
    if stock_option == "Select from popular stocks":
        stock_name = st.selectbox(
            "Select Stock",
            list(popular_stocks.keys()),
            index=0
        )
        stock = popular_stocks[stock_name]
    else:
        custom_stock = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS for NSE, TATASTEEL.BO for BSE)", 
                                   value="RELIANCE.NS")
        if custom_stock:
            stock = custom_stock.upper()
            stock_name = custom_stock
        else:
            st.warning("Please enter a stock symbol")

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())
    
    # Model selection
    selected_models = st.multiselect(
        "Select Models to Compare",
        ["Linear Regression", "LSTM", "GRU", "SimpleRNN", "ARIMA", "Prophet", "LSTM-GRU Hybrid"],
        default=["Linear Regression", "LSTM", "GRU"]
    )
    
    # Training parameters
    st.subheader("Model Parameters")
    epochs = st.slider("Epochs (for neural networks)", 1, 50, 10)
    batch_size = st.slider("Batch Size (for neural networks)", 16, 128, 32, step=16)
    sequence_length = st.slider("Sequence Length (for time series models)", 30, 200, 100, step=10)
    
    # Option to force retrain
    force_retrain = st.checkbox("Force Retrain Models", value=False, 
                                help="Check this to retrain models even if pre-trained versions exist")
    
    st.markdown("---")
    st.subheader("📂 Or Upload Your Data")
    st.subheader("Note: Data should have Date and Close Columns.")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    st.markdown("---")
    st.markdown("ℹ️ *Select models and adjust parameters to compare performance*")

# Check if end date is in the future
today = datetime.date.today()
future_prediction = end_date > today

# Initialize variables
data = None
use_uploaded_data = False

# Process uploaded file if provided
if uploaded_file is not None:
    try:
        data_load_state = st.info('📊 Loading custom data...')
        uploaded_data = pd.read_csv(uploaded_file)
        
        # Check required columns
        if 'Date' not in uploaded_data.columns or 'Close' not in uploaded_data.columns:
            st.error("Uploaded file must contain 'Date' and 'Close' columns")
            st.stop()
            
        uploaded_data['Date'] = pd.to_datetime(uploaded_data['Date'])
        uploaded_data['Close'] = pd.to_numeric(uploaded_data['Close'], errors='coerce')
        uploaded_data = uploaded_data.dropna(subset=['Close'])
        
        if len(uploaded_data) < sequence_length * 2:
            st.error(f"Not enough data points. Need at least {sequence_length * 2} records.")
            st.stop()
            
        data = uploaded_data.set_index('Date')[['Close']]
        data = data.sort_index()
        data_load_state.success('✅ Custom data loaded successfully!')
        use_uploaded_data = True
        stock_name_display = "Uploaded Stock Data"
        
        with st.expander("View uploaded data preview"):
            st.write(data.head())
            st.write(f"Data range: {data.index[0]} to {data.index[-1]}")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.stop()

# Function to cache downloaded data
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(stock, start_date, end_date):
    """Fetch stock data from Yahoo Finance with caching"""
    try:
        data = yf.download(stock, start=start_date, end=min(end_date, today))
        if data.empty:
            raise ValueError("No data returned from Yahoo Finance")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {stock}: {str(e)}")
        return None

# Download Stock Data if no file uploaded
if not use_uploaded_data:
    try:
        data_load_state = st.info(f'📊 Loading {stock} data...')
        data = fetch_stock_data(stock, start_date, end_date)
        
        if data is None or data.empty:
            st.error(f"Could not fetch data for {stock}. Please check the symbol is correct.")
            st.stop()
        
        data_load_state.success(f'✅ {stock} data loaded successfully!')
        stock_name_display = stock_name if stock_option == "Select from popular stocks" else stock
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Data validation
if len(data) < sequence_length * 2:
    st.error(f"Not enough data points for the selected sequence length. Need at least {sequence_length * 2} records.")
    st.info("Please try:")
    st.info("- Selecting a longer date range")
    st.info("- Reducing the sequence length")
    st.info("- Choosing a different stock")
    st.stop()

# Display raw data
with st.expander("🔍 View Raw Data", expanded=True):
    st.write(data)
    st.write("Most likely a stock price represents in USD (United States Dollars) — that's a common unit for stock prices.")

# Prepare Data
@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_data(data_close_values, sequence_length):
    """Preprocess data with caching"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close_values)
    
    # Train-Test Split
    train_size = int(len(scaled_data) * 0.80)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]  # Include last n days from train for sequence
    
    return scaled_data, train_data, test_data, scaler, train_size

# Create sequences with validation
def create_sequences(data, seq_length):
    """Create sequences with proper error handling"""
    if len(data) < seq_length:
        raise ValueError(f"Not enough data points. Need at least {seq_length} records but got {len(data)}")
    
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# For linear regression
def prepare_linear_data(data, seq_length):
    """Prepare data for linear regression"""
    if len(data) < seq_length:
        raise ValueError(f"Not enough data points. Need at least {seq_length} records but got {len(data)}")
    
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i].flatten())
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Get and preprocess data
try:
    data_close = data[['Close']].values
    data_hash = hashlib.md5(pd.util.hash_pandas_object(data[['Close']]).to_json().encode()).hexdigest()
    data_config_id = f"{data_hash}_{sequence_length}"
    
    scaled_data, train_data, test_data, scaler, train_size = preprocess_data(data_close, sequence_length)
    
    # Save scaler
    scaler_path = f"scalers/scaler_{data_config_id}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create sequences
    x_train, y_train = create_sequences(train_data, sequence_length)
    x_test, y_test = create_sequences(test_data, sequence_length)
    
    # For linear regression
    x_train_linear, y_train_linear = prepare_linear_data(train_data, sequence_length)
    x_test_linear, y_test_linear = prepare_linear_data(test_data, sequence_length)
    
except ValueError as e:
    st.error(f"Data processing error: {str(e)}")
    st.info("Please refresh the page and try with different parameters")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error during data processing: {str(e)}")
    st.stop()

# Model Building and Training
models = {}
model_performance = {}

def get_or_train_model(model_type, model_id):
    """Get or train a model with proper error handling"""
    model_path = f"models/{model_type}_{model_id}.h5"
    
    # Check if model exists and we're not forcing retrain
    if os.path.exists(model_path) and not force_retrain:
        try:
            st.info(f"⚡ Loading pre-trained {model_type} model...")
            start_time = time.time()
            if model_type == "Linear Regression":
                with open(model_path.replace('.h5', '.pkl'), 'rb') as f:
                    model = pickle.load(f)
            elif model_type in ["ARIMA", "Prophet"]:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = load_model(model_path)
            end_time = time.time()
            st.success(f"✅ {model_type} model loaded in {end_time - start_time:.2f} seconds")
            return model
        except Exception as e:
            st.warning(f"Failed to load pre-trained model: {str(e)}. Retraining...")
    
    # Train new model
    try:
        if model_type == "Linear Regression":
            with st.spinner(f'🧮 Training {model_type} Model...'):
                model = LinearRegression()
                model.fit(x_train_linear, y_train_linear)
                with open(model_path.replace('.h5', '.pkl'), 'wb') as f:
                    pickle.dump(model, f)
        
        elif model_type == "LSTM":
            with st.spinner(f'🧠 Training {model_type} Model...'):
                model = Sequential([
                    LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                    Dropout(0.2),
                    LSTM(100, return_sequences=True),
                    Dropout(0.2),
                    LSTM(100),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                model.save(model_path)
        
        elif model_type == "GRU":
            with st.spinner(f'🧠 Training {model_type} Model...'):
                model = Sequential([
                    GRU(100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                    Dropout(0.2),
                    GRU(100, return_sequences=True),
                    Dropout(0.2),
                    GRU(100),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                model.save(model_path)
        
        elif model_type == "SimpleRNN":
            with st.spinner(f'🧠 Training {model_type} Model...'):
                try:
                    # Initialize model
                    model = Sequential()
                    
                    # First layer with input shape
                    model.add(SimpleRNN(
                        units=100,
                        return_sequences=True,
                        input_shape=(x_train.shape[1], 1)
                    ))
                    model.add(Dropout(0.2))
                    
                    # Second layer
                    model.add(SimpleRNN(
                        units=100,
                        return_sequences=True
                    ))
                    model.add(Dropout(0.2))
                    
                    # Final RNN layer
                    model.add(SimpleRNN(units=100))
                    model.add(Dropout(0.2))
                    
                    # Output layer
                    model.add(Dense(units=1))
                    
                    # Compile model
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='mean_squared_error'
                    )
                    
                    # Train model
                    history = model.fit(
                        x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        validation_data=(x_test, y_test)
                    )
                    
                    # Save model
                    model.save(model_path)
                    st.success(f"✅ {model_type} model trained successfully!")
                    
                    return model
                    
                except Exception as e:
                    st.error(f"Failed to train {model_type} model: {str(e)}")
                    return None
        
        elif model_type == "ARIMA":
            with st.spinner(f'📊 Training {model_type} Model...'):
                # Check for stationarity
                result = adfuller(data_close.flatten())
                if result[1] > 0.05:
                    d = 1  # Data is not stationary, needs differencing
                else:
                    d = 0
                
                # Use auto_arima to find best parameters
                model = auto_arima(data_close, seasonal=False, trace=False,
                                  error_action='ignore', suppress_warnings=True,
                                  stepwise=True)
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        
        elif model_type == "Prophet":
            with st.spinner(f'📊 Training {model_type} Model...'):
                try:
                    # Prepare data for Prophet - must be a DataFrame with 'ds' and 'y' columns
                    prophet_data = data.reset_index()[['Date', 'Close']].copy()
                    prophet_data.columns = ['ds', 'y']  # Rename columns
                    
                    # Convert to proper dtypes
                    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                    prophet_data['y'] = pd.to_numeric(prophet_data['y'])
                    
                    # Remove any invalid values
                    prophet_data = prophet_data.dropna()
                    prophet_data = prophet_data[prophet_data['y'] > 0]
                    
                    if len(prophet_data) < 2:
                        raise ValueError("Not enough valid data points for Prophet")
                    
                    # Initialize model
                    model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.05
                    )
                    
                    # Fit model
                    model.fit(prophet_data)
                    
                    # Save model
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                        
                    return model
                    
                except Exception as e:
                    st.error(f"Prophet training failed: {str(e)}")
                    return None

        elif model_type == "LSTM-GRU Hybrid":
            with st.spinner(f'🧠 Training {model_type} Model...'):
                try:
                    # Hybrid model architecture
                    input_layer = Input(shape=(x_train.shape[1], 1))
                    
                    # LSTM branch
                    lstm_branch = LSTM(100, return_sequences=True)(input_layer)
                    lstm_branch = Dropout(0.2)(lstm_branch)
                    lstm_branch = LSTM(100, return_sequences=True)(lstm_branch)
                    lstm_branch = Dropout(0.2)(lstm_branch)
                    lstm_branch = LSTM(100)(lstm_branch)
                    lstm_branch = Dropout(0.2)(lstm_branch)
                    
                    # GRU branch
                    gru_branch = GRU(100, return_sequences=True)(input_layer)
                    gru_branch = Dropout(0.2)(gru_branch)
                    gru_branch = GRU(100, return_sequences=True)(gru_branch)
                    gru_branch = Dropout(0.2)(gru_branch)
                    gru_branch = GRU(100)(gru_branch)
                    gru_branch = Dropout(0.2)(gru_branch)
                    
                    # Combine branches
                    combined = concatenate([lstm_branch, gru_branch])
                    output = Dense(1)(combined)
                    
                    model = Model(inputs=input_layer, outputs=output)
                    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                    
                    # Add progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Custom training loop with progress updates
                    for epoch in range(epochs):
                        try:
                            history = model.fit(
                                x_train, 
                                y_train, 
                                epochs=1, 
                                batch_size=batch_size, 
                                verbose=0,
                                validation_data=(x_test, y_test)
                            )
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {history.history['loss'][0]:.4f}")
                        except Exception as e:
                            st.error(f"Error during epoch {epoch + 1}: {str(e)}")
                            continue
                    
                    # Save model only if training completed successfully
                    model.save(model_path)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ {model_type} model trained successfully!")
                    return model
                    
                except Exception as e:
                    st.error(f"Error training hybrid model: {str(e)}")
                    st.error("This might be due to insufficient memory or incompatible parameters. Try reducing batch size or sequence length.")
                    return None
        
        return model
    
    except Exception as e:
        st.error(f"Error training {model_type} model: {str(e)}")
        return None

# Get or train models - only if models are selected
if len(selected_models) > 0:
    for model_name in selected_models:
        model_id = generate_model_id(stock, start_date, end_date, model_name, sequence_length, epochs, batch_size)
        models[model_name] = get_or_train_model(model_name, model_id)
else:
    st.warning("Please select at least one model to proceed")
    st.stop()

# Remove any None models (that failed to train)
models = {k: v for k, v in models.items() if v is not None}
if not models:
    st.error("No models were successfully trained. Please check your parameters and try again.")
    st.stop()

# Make Predictions and Evaluate
try:
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Results Display
    st.markdown("---")
    st.markdown(f'<h2 class="stock-header">📉 {stock_name_display} Price Prediction Results</h2>', unsafe_allow_html=True)

    # Create columns for model cards
    cols = st.columns(len(models))
    model_colors = {
        "Linear Regression": "linear-reg-card",
        "LSTM": "lstm-card",
        "GRU": "gru-card",
        "SimpleRNN": "rnn-card",
        "ARIMA": "arima-card",
        "Prophet": "prophet-card",
        "LSTM-GRU Hybrid": "hybrid-card"
    }

    for idx, (model_name, model) in enumerate(models.items()):
        with cols[idx]:
            with st.container():
                card_class = model_colors.get(model_name, "model-card")
                st.markdown(f'<div class="model-card {card_class}">', unsafe_allow_html=True)
                st.subheader(model_name)
                
                try:
                    # Make predictions based on model type
                    if model_name == "Linear Regression":
                        # For Linear Regression we need to use the linear-prepared data
                        predictions = model.predict(x_test_linear)
                        predictions = predictions.reshape(-1, 1)
                    elif model_name in ["LSTM", "GRU", "SimpleRNN", "LSTM-GRU Hybrid"]:
                        # For neural networks we use the sequence data
                        predictions = model.predict(x_test)
                    elif model_name == "ARIMA":
                        # ARIMA returns predictions on original scale
                        predictions = model.predict(n_periods=len(y_test))
                        predictions = predictions.reshape(-1, 1)
                        # Scale to match other models' scale
                        predictions = scaler.transform(predictions)
                    elif model_name == "Prophet":
                        # Prophet needs date formatting
                        test_dates = data.index[train_size:train_size + len(y_test)]
                        future = pd.DataFrame({'ds': test_dates})
                        forecast = model.predict(future)
                        predictions = forecast['yhat'].values.reshape(-1, 1)
                        predictions = scaler.transform(predictions)
                    
                    # Ensure predictions are properly shaped
                    if predictions.ndim == 1:
                        predictions = predictions.reshape(-1, 1)
                    
                    # Inverse transform to get actual price values
                    predictions = scaler.inverse_transform(predictions)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test_rescaled, predictions)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test_rescaled, predictions)
                    mape = np.mean(np.abs((y_test_rescaled - predictions) / y_test_rescaled)) * 100
                    r2 = r2_score(y_test_rescaled, predictions)
                    accuracy = max(0, (1 - mape/100) * 100)
                    
                    # Store performance
                    model_performance[model_name] = {
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE": mape,
                        "R2": r2,
                        "Accuracy": accuracy,
                        "Predictions": predictions,
                        "Model": model
                    }
                    
                    # Display metrics
                    with st.expander("📊 View Metrics", expanded=True):
                        st.markdown(f"""
                        <div class="metric-box">
                        <b>MSE (Mean Squared Error):</b> {mse:.2f}<br>
                        <small>Average squared difference between actual and predicted values</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-box">
                        <b>RMSE (Root Mean Squared Error):</b> {rmse:.2f}<br>
                        <small>Square root of MSE, in same units as target</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-box">
                        <b>MAE (Mean Absolute Error):</b> {mae:.2f}<br>
                        <small>Average absolute difference between actual and predicted values</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-box">
                        <b>MAPE (Mean Absolute Percentage Error):</b> {mape:.2f}%<br>
                        <small>Percentage error between actual and predicted values</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-box">
                        <b>R² (R-squared Score):</b> {r2:.2f}<br>
                        <small>Proportion of variance explained by the model (0-1)</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="metric-box">
                        <b>Accuracy Score:</b> {accuracy:.2f}%<br>
                        <small>Percentage accuracy based on MAPE</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Prediction failed for {model_name}: {str(e)}")
                    continue
                
                st.markdown('</div>', unsafe_allow_html=True)

    # Plot Predictions
    test_dates = data.index[train_size:train_size + len(y_test_rescaled)]
    
    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_rescaled, color='blue', label='Actual Prices', linewidth=2)

    colors = ['#4b8df8', '#ff6b6b', '#51cf66', '#fcc419', '#845ef7', '#20c997', '#f76707']
    for idx, (model_name, perf) in enumerate(model_performance.items()):
        plt.plot(test_dates, perf["Predictions"], color=colors[idx], 
                label=f'{model_name} Predictions', linestyle='--')

    plt.title(f"{stock_name_display} Price Prediction Comparison", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price (USD)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    fig1.autofmt_xdate()
    st.pyplot(fig1)

    # Model-specific visualizations
    if "Prophet" in models:
        with st.expander("🔍 Prophet Model Components", expanded=True):
            st.write("Prophet decomposes the time series into components:")
            model = model_performance["Prophet"]["Model"]
            test_dates = data.index[train_size:train_size + len(y_test)]
            future = pd.DataFrame({'ds': test_dates})
            forecast = model.predict(future)
            
            fig_trend = model.plot_components(forecast)
            st.pyplot(fig_trend)
            
            st.write("""
            - **Trend**: Shows the overall direction of the stock price
            - **Weekly Seasonality**: Patterns that repeat weekly
            - **Yearly Seasonality**: Patterns that repeat yearly (if enabled)
            """)

    if "ARIMA" in models:
        with st.expander("🔍 ARIMA Model Summary", expanded=True):
            model = model_performance["ARIMA"]["Model"]
            st.write(f"ARIMA Model Order: {model.order}")
            st.write(f"ARIMA Model Seasonal Order: {model.seasonal_order if hasattr(model, 'seasonal_order') else 'None'}")
            st.write("""
            - **AR (p)**: Autoregressive terms (how many past values influence current value)
            - **I (d)**: Differencing terms (how many times differenced to make stationary)
            - **MA (q)**: Moving average terms (how many past errors influence current value)
            """)

    if "LSTM-GRU Hybrid" in models:
        with st.expander("🔍 Hybrid Model Architecture", expanded=True):
            st.write("""
            The LSTM-GRU Hybrid model combines the strengths of both architectures:
            
            - **LSTM Branch**: Better at learning long-term dependencies
            - **GRU Branch**: Faster training with comparable performance
            - **Combined**: Takes advantage of both architectures' strengths
            
            The model concatenates the outputs of both branches before making final predictions.
            """)

    if "Linear Regression" in selected_models:
        # Get the linear regression model
        lr_model = model_performance["Linear Regression"]["Model"]
            
        # Display the equation
        st.markdown("---")
        st.subheader("📈 Linear Regression Details")
            
        # Get coefficients (flattened features)
        coefficients = lr_model.coef_
        intercept = lr_model.intercept_
            
        # Create equation string
        equation = f"Predicted Price = {intercept:.4f}"
        for i, coef in enumerate(coefficients[0:5]):  # Show first 5 coefficients for brevity
            equation += f" + ({coef:.4f} * X{i+1})"
        if len(coefficients) > 5:
            equation += " + ..."
            
        st.markdown(f"""
            <div class="model-card linear-reg-card">
                <h3>Regression Equation</h3>
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px;">
                    <code>{equation}</code>
                </div>
                <p style="font-size: 0.9em; color: #666;">
                    Where X1-X{len(coefficients)} are the previous {sequence_length} days' prices
                </p>
            </div>
            """, unsafe_allow_html=True)

        fig_lr = plt.figure(figsize=(10, 6))
        
        # Plot actual vs predicted
        plt.scatter(y_test_rescaled, model_performance["Linear Regression"]["Predictions"], 
                    alpha=0.5, color='blue', label='Predictions')
            
        # Plot perfect prediction line
        max_val = max(y_test_rescaled.max(), model_performance["Linear Regression"]["Predictions"].max())
        min_val = min(y_test_rescaled.min(), model_performance["Linear Regression"]["Predictions"].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
        plt.title('Actual vs Predicted Prices (Linear Regression)')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        st.pyplot(fig_lr)

        if len(y_test_rescaled) > 1:  # Ensure we have enough data points
            # Calculate actual and predicted price directions (1=up, 0=flat, -1=down)
            actual_direction = np.sign(np.diff(y_test_rescaled.flatten()))
            pred_direction = np.sign(np.diff(model_performance["Linear Regression"]["Predictions"].flatten()))
            
            # Filter out zeros (flat periods) to make it binary classification (up/down)
            mask = (actual_direction != 0) & (pred_direction != 0)
            actual_direction = actual_direction[mask]
            pred_direction = pred_direction[mask]
            
            # Convert to binary (1=up, 0=down)
            actual_binary = (actual_direction > 0).astype(int)
            pred_binary = (pred_direction > 0).astype(int)
            
            # Create confusion matrix
            cm = confusion_matrix(actual_binary, pred_binary)

            # Create a smaller figure with adjusted parameters
            fig_cm = plt.figure(figsize=(3, 2), dpi=100)    # Smaller figure size (3x2 inches)

            # Create heatmap with smaller fonts and tighter layout
            ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Down', 'Up'],
                            yticklabels=['Down', 'Up'],
                            annot_kws={'size': 8},  # Smaller annotation font
                            cbar=False)  # Remove color bar to save space

            plt.title('Direction Prediction', fontsize=6, pad=2)
            plt.xlabel('Predicted', fontsize=6)
            plt.ylabel('Actual', fontsize=6)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)

            # Make the plot tighter
            plt.tight_layout()

            st.pyplot(fig_cm)
            
            # Calculate metrics
            total_predictions = len(actual_binary)
            correct_predictions = np.sum(actual_binary == pred_binary)
            direction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            st.markdown(f"""
            <div class="model-card" style="margin-top: 20px;">
                <h4>Direction Prediction Performance</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                    <div class="metric-box">
                        <b>Overall Direction Accuracy:</b> {direction_accuracy:.1%}
                    </div>
                    <div class="metric-box">
                        <b>Total Predictions:</b> {total_predictions}
                    </div>
                </div>
                <p style="font-size: 0.9em; margin-top: 10px;">
                    Note: Above 55% may be considered meaningful.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Not enough data points to calculate direction accuracy metrics")

    # Future Prediction Section
    if future_prediction:
        st.markdown("---")
        st.subheader("🔮 Future Price Predictions")
        
        days_ahead = (end_date - today).days
        st.info(f"Predicting {days_ahead} days into the future (from {today} to {end_date})")
        
        # Prepare future prediction data
        future_predictions = {}  # This will store ALL predictions for investment recommendations
        arima_prophet_predictions = {}
        other_predictions = {}
        
        for model_name, perf in model_performance.items():
            model = perf["Model"]
            future_preds = []
            
            if model_name == "Linear Regression":
                try:
                    current_sequence = scaled_data[-sequence_length:].flatten()
                    for _ in range(days_ahead):
                        pred = model.predict(current_sequence.reshape(1, -1))[0]
                        future_preds.append(pred)
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[-1] = pred
                    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                    other_predictions[model_name] = future_preds
                    future_predictions[model_name] = future_preds
                except Exception as e:
                    st.error(f"Error in Linear Regression future prediction: {str(e)}")
                    continue
                
            elif model_name in ["LSTM", "GRU", "SimpleRNN", "LSTM-GRU Hybrid"]:
                try:
                    current_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
                    for _ in range(days_ahead):
                        pred = model.predict(current_sequence)[0][0]
                        future_preds.append(pred)
                        current_sequence = np.roll(current_sequence, -1)
                        current_sequence[0, -1, 0] = pred
                    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                    other_predictions[model_name] = future_preds
                    future_predictions[model_name] = future_preds
                except Exception as e:
                    st.error(f"Error in {model_name} future prediction: {str(e)}")
                    continue
                
            elif model_name == "ARIMA":
                try:
                    future_preds = model.predict(n_periods=days_ahead)
                    future_preds = np.array(future_preds).reshape(-1, 1)
                    arima_prophet_predictions[model_name] = future_preds
                    future_predictions[model_name] = future_preds
                except Exception as e:
                    st.error(f"Error in ARIMA future prediction: {str(e)}")
                    continue
                
            elif model_name == "Prophet":
                try:
                    future_dates = pd.date_range(start=today + datetime.timedelta(days=1), end=end_date)
                    future_df = pd.DataFrame({'ds': future_dates})
                    forecast = model.predict(future_df)
                    future_preds = forecast['yhat'].values.reshape(-1, 1)
                    future_preds = scaler.inverse_transform(future_preds)
                    arima_prophet_predictions[model_name] = future_preds
                    future_predictions[model_name] = future_preds
                except Exception as e:
                    st.error(f"Error in Prophet future prediction: {str(e)}")
                    continue
            
        # Create date range for future predictions
        future_dates = pd.date_range(start=today + datetime.timedelta(days=1), end=end_date)
        
        # Create tabs for different model types
        tab1, tab2 = st.tabs(["Standard Models Forecast", "ARIMA & Prophet Forecast"])
        
        with tab1:
            if len(other_predictions) > 0:
                st.write("""
                    <h2 style='color:#2e86c1; text-align:center; font-family:Arial;'>
                        Standard Models Future Price Forecast
                    </h2>
                """, unsafe_allow_html=True)
                
                # Display predictions in a table
                future_df = pd.DataFrame(index=future_dates)
                for model_name, preds in other_predictions.items():
                    future_df[model_name] = preds
                
                styled_df = (future_df.style
                            .format("{:.2f}")
                            .set_properties(**{'background-color': '#f8f9f9', 
                                            'color': '#2c3e50',
                                            'border': '1px solid #ddd'})
                            .set_table_styles([{'selector': 'th',
                                                'props': [('background-color', '#2e86c1'),
                                                        ('color', 'white'),
                                                        ('font-weight', 'bold'),
                                                        ('text-align', 'center')]}]))

                st.dataframe(styled_df)
                
                # Plot predictions
                fig_standard = plt.figure(figsize=(14, 7))
                
                # Add last actual prices for context
                last_actual_dates = data.index[-30:]  # Last 30 days of actual data
                last_actual_prices = data['Close'][-30:]
                plt.plot(last_actual_dates, last_actual_prices, color='blue', 
                        label='Actual Prices', linewidth=2)
                
                # Plot all standard model predictions
                colors = ['#ff6b6b', '#51cf66', '#fcc419', '#4b8df8']
                for idx, (model_name, preds) in enumerate(other_predictions.items()):
                    plt.plot(future_dates, preds, color=colors[idx], 
                            label=f'{model_name} Forecast', linestyle='--')
                
                plt.title(f"{stock_name_display} Standard Models Forecast", fontsize=16)
                plt.xlabel("Date", fontsize=14)
                plt.ylabel("Price", fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                
                st.pyplot(fig_standard, use_container_width=True)
            else:
                st.warning("No future predictions available for standard models")
        
        with tab2:
            if len(arima_prophet_predictions) > 0:
                st.write("""
                    <h2 style='color:#2e86c1; text-align:center; font-family:Arial;'>
                        ARIMA & Prophet Future Price Forecast
                    </h2>
                """, unsafe_allow_html=True)
                
                # Display predictions in a table
                future_df = pd.DataFrame(index=future_dates)
                for model_name, preds in arima_prophet_predictions.items():
                    future_df[model_name] = preds
                
                styled_df = (future_df.style
                            .format("{:.2f}")
                            .set_properties(**{'background-color': '#f8f9f9', 
                                            'color': '#2c3e50',
                                            'border': '1px solid #ddd'})
                            .set_table_styles([{'selector': 'th',
                                                'props': [('background-color', '#2e86c1'),
                                                        ('color', 'white'),
                                                        ('font-weight', 'bold'),
                                                        ('text-align', 'center')]}]))

                st.dataframe(styled_df)
                
                # Plot predictions
                fig_arima_prophet = plt.figure(figsize=(14, 7))
                
                # Add last actual prices for context
                last_actual_dates = data.index[-30:]  # Last 30 days of actual data
                last_actual_prices = data['Close'][-30:]
                plt.plot(last_actual_dates, last_actual_prices, color='blue', 
                        label='Actual Prices', linewidth=2)
                
                # Plot ARIMA and Prophet predictions
                colors = ['#845ef7', '#20c997']  # Purple for ARIMA, teal for Prophet
                for idx, (model_name, preds) in enumerate(arima_prophet_predictions.items()):
                    plt.plot(future_dates, preds, color=colors[idx], 
                            label=f'{model_name} Forecast', linestyle='--')
                
                plt.title(f"{stock_name_display} ARIMA & Prophet Forecast", fontsize=16)
                plt.xlabel("Date", fontsize=14)
                plt.ylabel("Price", fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                
                st.pyplot(fig_arima_prophet, use_container_width=True)
            else:
                st.warning("No future predictions available for ARIMA/Prophet models")
            
        st.markdown("---")  # Add a horizontal line for separation

    # Model Comparison Chart
    st.markdown("---")
    st.subheader("📊 Model Performance Comparison Dashboard")

    if len(model_performance) > 1:
        # Custom styling
        st.markdown("""
        <style>
        .metric-box {
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .metric-box:hover {
            transform: translateY(-5px);
        }
        </style>
        """, unsafe_allow_html=True)

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Metrics Comparison", "📋 Detailed Scores", "📈 More Detailed Comparison"])
        
        with tab1:
            # Enhanced bar chart
            fig = plt.figure(figsize=(14, 8), facecolor='#f8f9fa')
            ax = fig.add_subplot(111)
            
            metrics = ['MSE', 'RMSE', 'MAPE']
            x = np.arange(len(metrics))
            width = 0.8 / len(model_performance)
            
            # Custom color palette
            colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
            
            for idx, (model_name, perf) in enumerate(model_performance.items()):
                values = [perf['MSE'], perf['RMSE'], perf['MAPE']]
                bars = ax.bar(x + idx*width, values, width, 
                            label=model_name, 
                            color=colors[idx % len(colors)],
                            edgecolor='white',
                            linewidth=1.5)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold')
            
            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Model Performance Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x + width*(len(model_performance)-1)/2)
            ax.set_xticklabels(metrics, fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            st.pyplot(fig)
            
            # Add interpretation
            with st.expander("💡 How to interpret these metrics", expanded=False):
                st.markdown("""
                - **MSE (Mean Squared Error)**: Lower is better (punishes large errors)
                - **RMSE (Root MSE)**: More interpretable in original units
                - **MAPE (Mean Absolute % Error)**: Percentage error (under 10% is excellent)
                """)
        
        with tab2:
            # Detailed metrics table with conditional formatting
            st.markdown("### 📊 Performance Metrics Table")
            
            # Prepare data
            metrics_data = []
            for model_name, perf in model_performance.items():
                metrics_data.append({
                    'Model': model_name,
                    'MSE': perf['MSE'],
                    'RMSE': perf['RMSE'],
                    'MAPE': f"{perf['MAPE']:.2f}%",
                    'R² Score': perf['R2'],
                    'Accuracy': f"{perf['Accuracy']:.2f}%"
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            
            # Apply styling
            def color_negative_red(val):
                color = 'red' if '%' in str(val) and float(val[:-1]) > 15 else 'green'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(
                df_metrics.style
                    .background_gradient(cmap='Blues', subset=['R² Score'])
                    .applymap(color_negative_red, subset=['MAPE'])
                    .format({'MSE': '{:.2f}', 'RMSE': '{:.2f}'})
                    .set_properties(**{'text-align': 'center'})
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('background', '#2a3f5f'), ('color', 'white')]
                    }]),
                use_container_width=True
            )
        
        with tab3:
            # Model performance comparison
            st.markdown("""
            <h3 style='color:#2e86c1; text-align:center; font-family:Arial;'>
                📊 Model Performance Comparison
            </h3>
            """, unsafe_allow_html=True)
            
            try:
                # Prepare data for visualization
                metrics = ['MSE', 'RMSE', 'MAPE', 'R2', 'Accuracy']
                models = list(model_performance.keys())
                
                # Normalize metrics for better visualization (lower is better for MSE/RMSE/MAPE)
                normalized_perf = {}
                for model in models:
                    normalized_perf[model] = {
                        'MSE': 1 - (model_performance[model]['MSE'] / max(p['MSE'] for p in model_performance.values())),
                        'RMSE': 1 - (model_performance[model]['RMSE'] / max(p['RMSE'] for p in model_performance.values())),
                        'MAPE': 1 - (model_performance[model]['MAPE'] / max(p['MAPE'] for p in model_performance.values())),
                        'R2': model_performance[model]['R2'],
                        'Accuracy': model_performance[model]['Accuracy']/100
                    }
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot each metric as a separate bar cluster
                bar_width = 0.15
                index = np.arange(len(metrics))
                
                for i, model in enumerate(models):
                    values = [normalized_perf[model][metric] for metric in metrics]
                    ax.bar(index + i*bar_width, values, bar_width, 
                        label=model, alpha=0.8)
                
                # Customize the plot
                ax.set_xlabel('Metrics', fontsize=12)
                ax.set_ylabel('Normalized Performance Score', fontsize=12)
                ax.set_title('Model Performance Across Different Metrics', fontsize=14, pad=20)
                ax.set_xticks(index + bar_width*1.5)
                ax.set_xticklabels(metrics)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels on top of bars
                for i, model in enumerate(models):
                    for j, metric in enumerate(metrics):
                        value = model_performance[model][metric]
                        if metric in ['R2', 'Accuracy']:
                            label = f"{value:.2f}" if metric == 'R2' else f"{value:.1f}%"
                        else:
                            label = f"{value:.4f}"
                        ax.text(j + i*bar_width, normalized_perf[model][metric] + 0.02, 
                            label, ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add explanation
                st.info("""
                **How to read this chart:**
                - For MSE, RMSE, and MAPE: Higher bars are better (values normalized to 0-1 scale)
                - For R² and Accuracy: Actual values shown (higher is better)
                - Exact values displayed above each bar
                """)
                
            except Exception as e:
                st.error(f"Error visualizing model performance: {str(e)}")

    # Add model recommendation
    best_model = min(model_performance.items(), key=lambda x: x[1]['MSE'])
    st.success(f"🏆 **Best Performing Model**: {best_model[0]} (MSE: {best_model[1]['MSE']:.2f})")

    # Performance summary cards
    cols = st.columns(len(model_performance))
    for idx, (model_name, perf) in enumerate(model_performance.items()):
        with cols[idx]:
            st.markdown(f"""
            <div class="metric-box" style="border-left: 5px solid {colors[idx]};">
                <h4>{model_name}</h4>
                <p>Accuracy: <strong>{perf['Accuracy']:.2f}%</strong></p>
                <p>MAPE: <strong>{perf['MAPE']:.2f}%</strong></p>
                <p>R²: <strong>{perf['R2']:.2f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    # Performance Summary
    st.markdown("---")
    st.subheader("🏆 Performance Summary")

    if len(model_performance) == 1:
        # Single model summary
        model_name, perf = next(iter(model_performance.items()))
        
        st.success(f"**Selected Model:** {model_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy Score", f"{perf['Accuracy']:.2f}%")
        
        with col2:
            st.metric("Mean Squared Error", f"{perf['MSE']:.2f}")
        
        with col3:
            st.metric("Mean Absolute Percentage Error", f"{perf['MAPE']:.2f}%")
        
        st.markdown(f"""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;">
            <h4>Interpretation:</h4>
            <ul>
                <li><b>Accuracy Score:</b> {perf['Accuracy']:.2f}% indicates the model's overall prediction accuracy</li>
                <li><b>MSE:</b> {perf['MSE']:.2f} suggests the average squared difference between predicted and actual values</li>
                <li><b>MAPE:</b> {perf['MAPE']:.2f}% shows the average percentage error in predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Multi-model comparison
        best_model = max(model_performance.items(), key=lambda x: x[1]["Accuracy"])
        worst_model = min(model_performance.items(), key=lambda x: x[1]["Accuracy"])

        # Sort Models by Accuracy
        sorted_models = sorted(model_performance.items(), key=lambda x: x[1]["Accuracy"], reverse=True)

        # Display results in columns
        cols = st.columns(len(model_performance))
        
        for idx, (model_name, perf) in enumerate(sorted_models):
            with cols[idx]:
                if idx == 0:
                    st.success(f"**Best Model:** {model_name}")
                    st.metric("Accuracy Score", f"{perf['Accuracy']:.2f}%", delta_color="off")
                    st.metric("MSE", f"{perf['MSE']:.2f}", delta_color="off")
                    st.metric("MAPE", f"{perf['MAPE']:.2f}%", delta_color="off")
                elif idx == len(sorted_models) - 1:
                    st.error(f"**Worst Model:** {model_name}")
                    st.metric("Accuracy Score", f"{perf['Accuracy']:.2f}%", delta_color="off")
                    st.metric("MSE", f"{perf['MSE']:.2f}", delta_color="off")
                    st.metric("MAPE", f"{perf['MAPE']:.2f}%", delta_color="off")
                else:
                    st.warning(f"**Moderate Model:** {model_name}")
                    st.metric("Accuracy Score", f"{perf['Accuracy']:.2f}%", delta_color="off")
                    st.metric("MSE", f"{perf['MSE']:.2f}", delta_color="off")
                    st.metric("MAPE", f"{perf['MAPE']:.2f}%", delta_color="off")

        # Explanation of results
        with st.expander("💡 Interpretation Guide", expanded=True):
            st.markdown("""
            - **Accuracy Score**: Percentage accuracy based on MAPE (higher is better)
            - **MSE (Mean Squared Error)**: Average squared difference between actual and predicted values (lower is better)
            - **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target (lower is better)
            - **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values (lower is better)
            - **MAPE (Mean Absolute Percentage Error)**: Percentage error between actual and predicted values (lower is better)
            - **R² (R-squared Score)**: Proportion of variance explained by the model (0-1, higher is better)
            
            *Note*: Linear Regression is simpler but often performs well for financial data.
            Neural networks (LSTM/GRU) may perform better with more data and tuning.
            """)

    # Moving Averages Plot
    st.markdown("---")
    st.subheader("📶 Technical Indicators")

    ma_periods = st.multiselect(
        "Select Moving Average Periods",
        [20, 50, 100, 200],
        default=[50, 100],
        key="ma_select"
    )

    if ma_periods:
        fig3 = plt.figure(figsize=(14, 7))
        plt.plot(data['Close'], label='Actual Prices', color='blue', linewidth=2)
        
        ma_colors = ['red', 'green', 'purple', 'orange']
        for idx, period in enumerate(ma_periods):
            ma = data['Close'].rolling(window=period).mean()
            plt.plot(ma, label=f'MA {period} Days', color=ma_colors[idx], linestyle='--')
        
        plt.title(f"{stock_name_display} Moving Averages", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)

    def get_investment_recommendation(predicted_growth, prediction_period_days, volatility):
        """
        Returns professional-grade recommendation based on:
        - predicted_growth: Percentage price change (e.g., 5.2 for 5.2%)
        - prediction_period_days: Horizon of prediction (e.g., 30 for 30-day forecast)
        - volatility: Annualized volatility (e.g., 0.25 for 25%)
        """
        # Ensure inputs are scalars
        predicted_growth = float(predicted_growth)
        prediction_period_days = int(prediction_period_days)
        volatility = float(volatility)

        # Annualize the predicted growth
        annualized_growth = ((1 + predicted_growth / 100) ** (365 / prediction_period_days) - 1) * 100

        # Risk-adjusted metric (Simplified Sharpe Ratio)
        risk_adjusted = annualized_growth / (volatility * 100) if volatility > 0 else 0

        # Professional Thresholds
        if prediction_period_days <= 30:  # Short-term
            if predicted_growth > 5 and risk_adjusted > 1.5:
                return "STRONG BUY (High Momentum)"
            elif predicted_growth > 3:
                return "BUY (Positive Trend)"
            elif -2 <= predicted_growth <= 2:
                return "HOLD (Neutral)"
            elif predicted_growth < -3:
                return "STRONG SELL (Downtrend)"
            else:
                return "SELL (Weakness)"

        elif prediction_period_days <= 365:  # Medium-term
            if annualized_growth > 20 and risk_adjusted > 1.2:
                return "STRONG BUY (Undervalued)"
            elif annualized_growth > 12:
                return "BUY (Growth Potential)"
            elif 5 <= annualized_growth <= 12:
                return "HOLD (Market Returns)"
            else:
                return "SELL (Underperforming)"

        else:  # Long-term
            if annualized_growth > 15:
                return "STRONG BUY (Compounding Opportunity)"
            elif annualized_growth > 10:
                return "BUY (Long-Term Growth)"
            elif 5 <= annualized_growth <= 10:
                return "HOLD (Market Matching)"
            else:
                return "SELL (Reallocate Capital)"

    def calculate_volatility(data, window=30):
        """Calculate annualized volatility from price data"""
        if len(data) < 2:
            return 0.25  # Default reasonable volatility if not enough data
        
        try:
            returns = np.log(data['Close'] / data['Close'].shift(1))
            rolling_std = returns.rolling(window=min(window, len(returns))).std()
            annualized_vol = rolling_std.iloc[-1] * np.sqrt(252)  # Last available value
            return float(annualized_vol) if not np.isnan(annualized_vol) else 0.25
        except Exception:
            return 0.25  # Fallback value

    # Investment recommendation logic
    if future_prediction and len(model_performance) > 0:
        st.markdown("---")
        st.subheader("💼 Professional Investment Recommendation")
    
        try:
            # Calculate volatility
            volatility = calculate_volatility(data)
            
            # Get predictions from all models (using future_predictions which contains all models)
            recommendations = []
            for model_name, preds in future_predictions.items():
                if len(preds) > 0:
                    initial_price = float(preds[0][0])
                    final_price = float(preds[-1][0])
                    growth = ((final_price - initial_price) / initial_price) * 100
                    days_ahead = (end_date - today).days
                    
                    if not np.isnan(growth) and days_ahead > 0:
                        rec = get_investment_recommendation(
                            float(growth), 
                            int(days_ahead), 
                            float(volatility)
                        )
                        recommendations.append((model_name, growth, rec))
            
            # Display recommendations if we have any
            if recommendations:
                # Display professional analysis
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Market Volatility", f"{volatility*100:.1f}%", 
                            help="Annualized volatility (30-day historical)")
                    
                with col2:
                    st.metric("Risk-Free Rate", "4.5-5.5%", 
                            help="Current 10-year Treasury yield range")
                
                # Consensus recommendation
                st.markdown("#### 📊 Consensus Analysis")
                
                # Create DataFrame for display
                rec_df = pd.DataFrame(recommendations, 
                                    columns=["Model", "Predicted Growth", "Recommendation"])
                
                # Color coding
                def color_recommendation(val):
                    if isinstance(val, str):
                        color = "green" if "BUY" in val else (
                            "red" if "SELL" in val else "orange")
                        return f"color: {color}; font-weight: bold"
                    return ""
                
                st.dataframe(
                    rec_df.style.format({"Predicted Growth": "{:.2f}%"})
                            .applymap(color_recommendation, subset=["Recommendation"])
                )
                
                # Add professional disclaimer
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <h4>📌 Professional Guidelines</h4>
                    <ul>
                        <li><b>STRONG BUY</b>: >5% short-term or >20% annualized growth with low volatility</li>
                        <li><b>BUY</b>: >3% short-term or >12% annualized growth</li>
                        <li><b>HOLD</b>: Neutral outlook (-2% to +2% short-term)</li>
                        <li><b>SELL</b>: <-3% short-term or <5% annualized growth</li>
                    </ul>
                    <p style="font-size: 0.9em; color: #666;">
                        Note: Recommendations incorporate volatility-adjusted metrics. 
                        Always verify with fundamental analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Could not generate investment recommendations for any models")
        except Exception as e:
            st.error(f"Error generating investment recommendations: {str(e)}")
            
    else:
        st.warning("Could not calculate investment recommendation. Please select the commence date for Prediction.")
        

except Exception as e:
    if len(selected_models) > 5:
        st.info("Please select only 5 models for getting Investment Recommendation.")
   
    else:
        st.info("Please refresh the page and try again with different parameters")
    st.stop()

finally:
    # Download Section
    st.markdown("---")
    st.subheader(f"📥 Download Dataset of {stock_name_display}")

    csv_data = data.to_csv().encode('utf-8')
    st.download_button(
            label="Download Stock Data as CSV",
            data=csv_data,
            file_name=f"{stock_name_display.replace(' ', '_')}_stock_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p>Developed using Streamlit, Keras, and Yahoo Finance</p>
            <p>ℹ️ Note: Stock predictions are for educational purposes only</p>
        </div>
        """, unsafe_allow_html=True)
