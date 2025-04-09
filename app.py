import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st
import datetime
from PIL import Image
import base64

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
    stock_name = st.selectbox(
        "Select Stock",
        list(popular_stocks.keys()),
        index=0
    )
    stock = popular_stocks[stock_name]
    
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
    if len(selected_models) > 3:
        st.warning("Please select Max. 3 models! for Comparison.")
    
    # Training parameters
    st.subheader("Model Parameters")
    epochs = st.slider("Epochs (for neural networks)", 1, 50, 10)
    batch_size = st.slider("Batch Size (for neural networks)", 16, 128, 32, step=16)
    sequence_length = st.slider("Sequence Length (for time series models)", 30, 200, 100, step=10)
    
    st.markdown("---")
    st.markdown("ℹ️ *Select models and adjust parameters to compare performance*")

# Check if end date is in the future
today = datetime.date.today()
future_prediction = end_date > today

# Download Stock Data
@st.cache_data
def load_data(stock, start, end):
    return yf.download(stock, start=start, end=end)

data_load_state = st.info(f'📊 Loading {stock_name} ({stock}) data...')
data = load_data(stock, start_date, min(end_date, today))  # Don't try to download future data
data_load_state.success(f'✅ {stock_name} ({stock}) data loaded successfully!')

# Display raw data
with st.expander("🔍 View Raw Data", expanded=True):
    st.write(data)
    st.write("Most likely a stock price respresents in USD (United States Dollars) — that's a common unit for stock prices.")

# Prepare Data
data_close = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
if len(data_close) > 0:
    scaled_data = scaler.fit_transform(data_close)

    # Train-Test Split
    train_size = int(len(scaled_data) * 0.80)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]  # Include last n days from train for sequence

    # Prepare Sequences
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(seq_length, len(data)):
            x.append(data[i-seq_length:i])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    # For neural network models
    x_train, y_train = create_sequences(train_data, sequence_length)
    x_test, y_test = create_sequences(test_data, sequence_length)

    # For linear regression (we'll use a simpler approach)
    def prepare_linear_data(data, seq_length):
        x, y = [], []
        for i in range(seq_length, len(data)):
            x.append(data[i-seq_length:i].flatten())  # Flatten the sequence
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    x_train_linear, y_train_linear = prepare_linear_data(train_data, sequence_length)
    x_test_linear, y_test_linear = prepare_linear_data(test_data, sequence_length)

    # Model Building and Training
    models = {}
    model_performance = {}

    # Linear Regression Model
    if "Linear Regression" in selected_models:
        with st.spinner('🧮 Training Linear Regression Model...'):
            linear_model = LinearRegression()
            linear_model.fit(x_train_linear, y_train_linear)
            models["Linear Regression"] = linear_model

    # LSTM Model
    if "LSTM" in selected_models:
        with st.spinner('🧠 Training LSTM Model...'):
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                LSTM(50, return_sequences=True),
                LSTM(50),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            models["LSTM"] = lstm_model

    # GRU Model
    if "GRU" in selected_models:
        with st.spinner('🧠 Training GRU Model...'):
            gru_model = Sequential([
                GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                GRU(50, return_sequences=True),
                GRU(50),
                Dense(1)
            ])
            gru_model.compile(optimizer='adam', loss='mean_squared_error')
            gru_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            models["GRU"] = gru_model

    # SimpleRNN Model
    if "SimpleRNN" in selected_models:
        with st.spinner('🧠 Training SimpleRNN Model...'):
            rnn_model = Sequential([
                SimpleRNN(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                SimpleRNN(50, return_sequences=True),
                SimpleRNN(50),
                Dense(1)
            ])
            rnn_model.compile(optimizer='adam', loss='mean_squared_error')
            rnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            models["SimpleRNN"] = rnn_model

    # Make Predictions and Evaluate
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Results Display
    st.markdown("---")
    st.markdown(f'<h2 class="stock-header">📉 {stock_name} ({stock}) Price Prediction Results</h2>', unsafe_allow_html=True)

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
                accuracy = max(0, (1 - mape/100) * 100)  # Simple accuracy metric
                
                st.subheader(model_name)
                
                # Display metrics with full forms
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
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store performance for comparison
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

    # Plot Predictions
    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(y_test_rescaled, color='blue', label='Actual Prices', linewidth=2)

    colors = ['#4b8df8', '#ff6b6b', '#51cf66', '#fcc419']
    for idx, (model_name, perf) in enumerate(model_performance.items()):
        plt.plot(perf["Predictions"], color=colors[idx], label=f'{model_name} Predictions', linestyle='--')

    plt.title(f"{stock_name} Stock Price Prediction Comparison", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig1)

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
        
        plt.title(f"{stock_name} Future Price Forecast", fontsize=16)
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
        
        plt.title(f"{stock_name} Moving Averages", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Price", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)

    # Model Comparison Chart (only if more than one model selected)
    if len(model_performance) > 1:
        st.markdown("---")
        st.subheader("📊 Model Performance Comparison")
        
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Accuracy', 'MSE', 'RMSE', 'MAPE']
        x = np.arange(len(metrics))
        width = 0.8 / len(model_performance)
        
        for idx, (model_name, perf) in enumerate(model_performance.items()):
            values = [perf['Accuracy'], perf['MSE'], perf['RMSE'], perf['MAPE']]
            ax.bar(x + idx*width, values, width, label=model_name, color=colors[idx])
        
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison by Different Metrics')
        ax.set_xticks(x + width*(len(model_performance)-1)/2)
        ax.set_xticklabels(['Accuracy (%)', 'MSE', 'RMSE', 'MAPE (%)'])
        ax.legend()
        
        st.pyplot(fig2)

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
            "Avg Growth (%)": ["+6.5%", "+2.1%", "-4.3%"],
            "Recommendation": ["BUY", "HOLD", "SELL"],
            "Reasoning": [
                "Strong potential for appreciation",
                "Modest growth expected",
                "Predicted loss over forecast period"
            ]
            }

        df = pd.DataFrame(data1)
        st.markdown("### 💼 Interpretation Guide")
        st.dataframe(
            df.style.set_properties(**{
                'text-align': 'center',
                'font-size': '16px'
            }).hide(axis="index"),
            use_container_width=True
            )
    else:
        st.warning("Could not calculate investment recommendation - no valid predictions available")
    

# Download Section
    st.markdown("---")
    st.subheader("📥 Download Results")

    csv_data = data.to_csv().encode('utf-8')
    st.download_button(
        label="Download Stock Data as CSV",
        data=csv_data,
        file_name=f"{stock}_stock_data.csv",
        mime="text/csv"
    )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p>Developed with ❤️ using Streamlit, Keras, and Yahoo Finance</p>
            <p>ℹ️ Note: Stock predictions are for educational purposes only</p>
        </div>
        """, unsafe_allow_html=True)
else:
    if len(data['Close']) == 0:
        st.error("Change date with Yesterday's date!")
    st.write(f"Due to some technical issues stock data of a {stock_name} is not available so please select previous dates ", unsafe_allow_html=True )
