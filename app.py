# All Required Imports
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
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
        ["Linear Regression", "LSTM", "GRU", "SimpleRNN"],
        default=["Linear Regression", "LSTM", "GRU"]
    )
    if len(selected_models) == 0:
        st.warning("Please select at least 1 Model for Processing!")
    if len(selected_models) > 3:
        st.warning("Please select Max. 3 models! for Comparison.")
    
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
                    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                    LSTM(50, return_sequences=True),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                model.save(model_path)
        
        elif model_type == "GRU":
            with st.spinner(f'🧠 Training {model_type} Model...'):
                model = Sequential([
                    GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                    GRU(50, return_sequences=True),
                    GRU(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                model.save(model_path)
        
        elif model_type == "SimpleRNN":
            with st.spinner(f'🧠 Training {model_type} Model...'):
                model = Sequential([
                    SimpleRNN(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                    SimpleRNN(50, return_sequences=True),
                    SimpleRNN(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                model.save(model_path)
        
        return model
    
    except Exception as e:
        st.error(f"Error training {model_type} model: {str(e)}")
        return None

# Get or train models
for model_name in selected_models:
    model_id = generate_model_id(stock, start_date, end_date, model_name, sequence_length, epochs, batch_size)
    models[model_name] = get_or_train_model(model_name, model_id)

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
        "SimpleRNN": "rnn-card"
    }

    for idx, (model_name, model) in enumerate(models.items()):
        with cols[idx]:
            with st.container():
                card_class = model_colors.get(model_name, "model-card")
                st.markdown(f'<div class="model-card {card_class}">', unsafe_allow_html=True)
                st.subheader(model_name)
                
                try:
                    # Make predictions
                    if model_name == "Linear Regression":
                        predictions = model.predict(x_test_linear)
                    else:
                        predictions = model.predict(x_test)
                    
                    predictions = predictions.reshape(-1, 1)
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
                
                except IndexError as e:
                    if "pop from empty list" in str(e):
                        st.error(f"⚠️ Prediction failed for {model_name}: Not enough data available")
                        st.info("Please try refreshing the page and selecting different parameters")
                    else:
                        st.error(f"Prediction failed for {model_name}: {str(e)}")
                    continue
                
                except Exception as e:
                    st.error(f"Prediction failed for {model_name}: {str(e)}")
                    continue
                
                st.markdown('</div>', unsafe_allow_html=True)

    # Plot Predictions
    test_dates = data.index[train_size:train_size + len(y_test_rescaled)]
    
    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test_rescaled, color='blue', label='Actual Prices', linewidth=2)

    colors = ['#4b8df8', '#ff6b6b', '#51cf66', '#fcc419']
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


        actual_direction = np.sign(np.diff(y_test_rescaled.flatten()))
        pred_direction = np.sign(np.diff(model_performance["Linear Regression"]["Predictions"].flatten()))
        
        # Create confusion matrix
        
        cm = confusion_matrix(actual_direction, pred_direction)

        # Create a smaller figure with adjusted parameters
        fig_cm = plt.figure(figsize=(3, 2), dpi=100)  # Smaller figure size (3x2 inches)

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
        
        if len(y_test_rescaled) > 1:  # Ensure we have enough data points
        # Calculate actual and predicted price directions (1=up, 0=flat, -1=down)
            actual_changes = np.diff(y_test_rescaled.flatten())
            pred_changes = np.diff(model_performance["Linear Regression"]["Predictions"].flatten())
            
            # Handle flat movements (you might want to count these separately)
            actual_direction = np.sign(actual_changes)
            pred_direction = np.sign(pred_changes)
            
            # Create confusion matrix (values will be -1, 0, 1)
            cm = confusion_matrix(actual_direction, pred_direction, labels=[-1, 0, 1])
            
            # Calculate metrics
            total_predictions = len(actual_direction)
            correct_predictions = np.sum(actual_direction == pred_direction)
            direction_accuracy = correct_predictions / total_predictions
            
            # Calculate up/down accuracy (excluding flat movements)
            up_actual = (actual_direction == 1)
            down_actual = (actual_direction == -1)
            
            up_correct = np.sum((pred_direction == 1) & up_actual) / np.sum(up_actual) if np.sum(up_actual) > 0 else 0
            down_correct = np.sum((pred_direction == -1) & down_actual) / np.sum(down_actual) if np.sum(down_actual) > 0 else 0
            
            st.markdown(f"""
            <div class="model-card" style="margin-top: 20px;">
                <h4>Direction Prediction Performance</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                    <div class="metric-box">
                        <b>Overall Direction Accuracy:</b> {direction_accuracy:.1%}
                    </div>
                    <div class="metric-box">
                        <b>Up Moves Correctly Predicted:</b> {up_correct:.1%}
                    </div>
                    <div class="metric-box">
                        <b>Down Moves Correctly Predicted:</b> {down_correct:.1%}
                    </div>
                </div>
                <p style="font-size: 0.9em; margin-top: 10px;">
                    Note: Random guessing would yield ~50% accuracy. Above 55% may be considered meaningful.
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
            last_sequence = scaled_data[-sequence_length:]
            future_predictions = {}
            
            for model_name, perf in model_performance.items():
                model = perf["Model"]
                future_preds = []
                current_sequence = last_sequence.copy()
                
                for _ in range(days_ahead):
                    if model_name == "Linear Regression":
                        pred = model.predict(current_sequence.flatten().reshape(1, -1))[0]
                    else:
                        pred = model.predict(current_sequence.reshape(1, sequence_length, 1))[0][0]
                    
                    future_preds.append(pred)
                    
                    # Update sequence
                    current_sequence = np.roll(current_sequence, -1)
                    current_sequence[-1] = pred
                    
                # Rescale predictions
                future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
                future_predictions[model_name] = future_preds
            
            # Create date range for future predictions
            future_dates = pd.date_range(start=today + datetime.timedelta(days=1), end=end_date)
            
            # Display future predictions in a table
            future_df = pd.DataFrame(index=future_dates)
            for model_name, preds in future_predictions.items():
                future_df[model_name] = preds
            
            st.write("Predicted Prices:")
            st.dataframe(future_df.style.format("{:.2f}"))
            
            # Plot future predictions
            fig_future = plt.figure(figsize=(14, 7))
            for idx, (model_name, preds) in enumerate(future_predictions.items()):
                plt.plot(future_dates, preds, color=colors[idx], label=f'{model_name} Forecast', linestyle='--')
            
            # Add last actual prices for context
            last_actual_dates = data.index[-30:]  # Last 30 days of actual data
            last_actual_prices = data['Close'][-30:]
            plt.plot(last_actual_dates, last_actual_prices, color='blue', label='Actual Prices', linewidth=2)
            
            plt.title(f"{stock_name_display} Future Price Forecast", fontsize=16)
            plt.xlabel("Date", fontsize=14)
            plt.ylabel("Price", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            st.pyplot(fig_future)

        # Model Comparison Chart
        st.markdown("---")
        st.subheader("📊 Model Performance Comparison")

        if len(model_performance) > 1:
            fig2, ax = plt.subplots(figsize=(12, 6))
            
            metrics = ['MSE', 'RMSE', 'MAPE']
            x = np.arange(len(metrics))
            width = 0.8 / len(model_performance)
            
            for idx, (model_name, perf) in enumerate(model_performance.items()):
                values = [perf['MSE'], perf['RMSE'], perf['MAPE']]
                ax.bar(x + idx*width, values, width, label=model_name, color=colors[idx])
            
            ax.set_ylabel('Score')
            ax.set_title('Model Comparison by Different Metrics')
            ax.set_xticks(x + width*(len(model_performance)-1)/2)
            ax.set_xticklabels(metrics)
            ax.legend()
            
            st.pyplot(fig2)

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
            best_model = max(model_performance.items(), key=lambda x: x[1]["Accuracy"])
            worst_model = min(model_performance.items(), key=lambda x: x[1]["Accuracy"])

            # Sort Models by Accuracy
            sorted_models = sorted(model_performance.items(), key=lambda x: x[1]["Accuracy"])

            # Get the Moderate Model (Middle One)
            Moderate_model = sorted_models[len(sorted_models) // 2] if len(sorted_models) > 2 else None

            # Display Results in Three Columns
            col1, col2, col3 = st.columns(3)
                
            with col1:
                st.success(f"**Best Model:** {best_model[0]}")
                st.metric("Accuracy Score", f"{best_model[1]['Accuracy']:.2f}%", delta_color="off")
                st.metric("MSE (Mean Squared Error)", f"{best_model[1]['MSE']:.2f}", delta_color="off")
                st.metric("MAPE (Mean Absolute Percentage Error)", f"{best_model[1]['MAPE']:.2f}%", delta_color="off")
            
            if Moderate_model:
                with col2:
                    st.success(f"**Moderate Model:** {Moderate_model[0]}")
                    st.metric("Accuracy Score", f"{Moderate_model[1]['Accuracy']:.2f}%", delta_color="off")
                    st.metric("MSE (Mean Squared Error)", f"{Moderate_model[1]['MSE']:.2f}", delta_color="off")
                    st.metric("MAPE (Mean Absolute Percentage Error)", f"{Moderate_model[1]['MAPE']:.2f}%", delta_color="off")
            
            with col3:
                st.error(f"**Worst Model:** {worst_model[0]}")
                st.metric("Accuracy Score", f"{worst_model[1]['Accuracy']:.2f}%", delta_color="off")
                st.metric("MSE (Mean Squared Error)", f"{worst_model[1]['MSE']:.2f}", delta_color="off")
                st.metric("MAPE (Mean Absolute Percentage Error)", f"{worst_model[1]['MAPE']:.2f}%", delta_color="off")


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
        
        if future_prediction and len([m for m in models.keys() if m != "GARCH"]) > 0:
            st.markdown("---")
            st.subheader("💼 Investment Recommendation")
                
            # Calculate average predicted growth across all models
            avg_growth_percent = []
            for model_name, preds in future_predictions.items():
                initial_price = preds[0][0]
                final_price = preds[-1][0]
                growth = ((final_price - initial_price) / initial_price) * 100
                avg_growth_percent.append(growth)
                
            if len(avg_growth_percent) > 0:
                avg_growth = np.mean(avg_growth_percent)
                max_growth = np.max(avg_growth_percent)
                min_growth = np.min(avg_growth_percent)
                    
                # Determine recommendation
                if avg_growth > 5:
                    recommendation = "BUY"
                    recommendation_color = "green"
                    reasoning = f"The average predicted growth across models is {avg_growth:.2f}%, suggesting strong potential for appreciation."
                elif avg_growth > 0:
                    recommendation = "HOLD"
                    recommendation_color = "orange"
                    reasoning = f"The average predicted growth across models is {avg_growth:.2f}%, suggesting modest potential for appreciation."
                else:
                    recommendation = "SELL"
                    recommendation_color = "red"
                    reasoning = f"The average predicted growth across models is {avg_growth:.2f}%, suggesting potential for depreciation."
                    
                # Display recommendation
                st.markdown(f"""
                <div style="border-left: 5px solid {recommendation_color}; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                    <h3 style="color: {recommendation_color};">Recommendation: <strong>{recommendation}</strong></h3>
                    <p>{reasoning}</p>
                </div>
                """, unsafe_allow_html=True)
                    
                # Display growth metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Predicted Growth", f"{avg_growth:.2f}%")
                with col2:
                    st.metric("Maximum Model Prediction", f"{max_growth:.2f}%")
                with col3:
                    st.metric("Minimum Model Prediction", f"{min_growth:.2f}%")
                    
                # Additional analysis
                with st.expander("📈 Detailed Analysis", expanded=True):
                    st.markdown("""
                    **Considerations for this recommendation:**
                    - Based on average predicted price change over the forecast period
                    - Incorporates predictions from all selected models
                    - More reliable when multiple models agree on direction
                        
                    **Important Notes:**
                    - This is not financial advice
                    - Consider other factors like company fundamentals, market conditions, and your risk tolerance
                    - Past performance is not indicative of future results
                    - Diversify your investments
                    """)
                        
                    # Show model-by-model growth predictions
                    st.write("Model-specific growth predictions:")
                    growth_data = []
                    for model_name, preds in future_predictions.items():
                        initial = preds[0][0]
                        final = preds[-1][0]
                        growth = ((final - initial) / initial) * 100
                        growth_data.append([model_name, initial, final, growth])
                        
                    growth_df = pd.DataFrame(growth_data, columns=["Model", "Initial Price", "Final Price", "Growth %"])
                    st.dataframe(growth_df.style.format({
                        "Initial Price": "{:.2f}",
                        "Final Price": "{:.2f}",
                        "Growth %": "{:.2f}%"
                    }))
            
            data1 = {
                "Recommendation": ["BUY", "HOLD", "SELL"],
                "Avg Growth (%)": ["+6.5% or more", "+2.1% to +6.4%", "-4.3% or less"],
                "Reasoning": [
                    "Strong potential for appreciation",
                    "Modest growth expected",
                    "Predicted loss over forecast period"
                ]
            }

            df = pd.DataFrame(data1)

            st.markdown("""
            ### 💼 Interpretation Guide (Standard Thresholds)
                        
            **These are general recommendation thresholds, not your actual predictions:**  
            *Your specific recommendation appears above based on the model's forecast.*
            """)

            st.dataframe(
                df.style.set_properties(**{
                    'text-align': 'center',
                    'font-size': '16px',
                    'background-color': '#f8f9fa'  # Light gray background
                }).hide(axis="index"),
                use_container_width=True
            )

            # Add explanatory note
            st.caption("""
            ℹ️ Note: These thresholds are illustrative. Actual investment decisions should consider 
            additional factors like market conditions, company fundamentals, and risk tolerance.
            """)
        else:
            st.warning("Could not calculate investment recommendation. Please select the commence date for Prediction.")
            
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
        
    else:
        if len(data) == 0:
            st.error("Change date with Yesterday's date!")
            st.write(f"Due to some technical issues stock data of a {stock_name} is not available so please select previous dates")
        if len(selected_models) == 0:
            st.write("Please select at least one model for processing!")

except Exception as e:
    st.error(f"An error occurred during prediction: {str(e)}")
    st.info("Please refresh the page and try again with different parameters")
    st.stop()

# [Include all the remaining sections from your original code here]
# Make sure to maintain proper error handling throughout

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>Developed using Streamlit, Keras, and Yahoo Finance</p>
        <p>ℹ️ Note: Stock predictions are for educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)