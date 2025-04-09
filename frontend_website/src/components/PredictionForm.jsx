import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Grid, Paper, InputAdornment, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, CircularProgress } from '@mui/material';
import { styled } from '@mui/system';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import * as tf from '@tensorflow/tfjs';

const FormPaper = styled(Paper)({
  padding: '2rem',
  borderRadius: '12px',
  boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
});

const PredictButton = styled(Button)({
  padding: '12px 24px',
  fontSize: '1rem',
  fontWeight: '600',
  borderRadius: '8px',
  textTransform: 'none',
  marginTop: '1rem',
  width: '100%',
});

const PredictionForm = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(() => {
    const date = new Date();
    date.setDate(date.getDay() + 7);
    return date;
  });
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);

  const fetchHistoricalData = async (symbol) => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(
        `https://api.allorigins.win/get?url=${encodeURIComponent(
          `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1y`
        )}`
      );
      
      const data = await response.json();
      const parsedData = JSON.parse(data.contents);
      
      if (!parsedData.chart || !parsedData.chart.result) {
        throw new Error('Invalid data format from Yahoo Finance');
      }
      
      const timestamps = parsedData.chart.result[0].timestamp;
      const closes = parsedData.chart.result[0].indicators.quote[0].close;
      
      const formattedData = timestamps.map((timestamp, index) => ({
        date: new Date(timestamp * 1000),
        price: closes[index]
      })).filter(item => item.price !== null);
      
      setHistoricalData(formattedData);
      return formattedData;
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to fetch historical data. Please try again.');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const trainAndPredict = async (data, predictionDays) => {
    try {
      setIsLoading(true);
      
      const prices = data.map(item => item.price);
      const days = data.map((_, index) => index);
      
      const xs = tf.tensor1d(days);
      const ys = tf.tensor1d(prices);
      
      const xMin = xs.min();
      const xMax = xs.max();
      const yMin = ys.min();
      const yMax = ys.max();
      
      const xsNorm = xs.sub(xMin).div(xMax.sub(xMin));
      const ysNorm = ys.sub(yMin).div(yMax.sub(yMin));
      
      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
      
      model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError'
      });
      
      await model.fit(xsNorm, ysNorm, {
        epochs: 100,
        batchSize: 32,
        verbose: 0
      });
      
      const predictionDates = [];
      const predictionResults = [];
      
      for (let i = 0; i < predictionDays; i++) {
        const futureDay = days.length + i;
        const normalizedDay = tf.tensor1d([futureDay]).sub(xMin).div(xMax.sub(xMin));
        const prediction = model.predict(normalizedDay);
        const denormalizedPred = prediction.mul(yMax.sub(yMin)).add(yMin);
        const predValue = (await denormalizedPred.data())[0];
        
        const predictionDate = new Date(startDate);
        predictionDate.setDate(predictionDate.getDate() + i);
        
        predictionDates.push(predictionDate);
        predictionResults.push(predValue);

        tf.dispose([normalizedDay, prediction, denormalizedPred]);
      }
      
      const predictions = predictionDates.map((date, index) => ({
        date,
        predictedPrice: predictionResults[index]
      }));
      
      const lastActualPrice = prices[prices.length - 1];
      const firstPredictedPrice = predictionResults[0];
      const priceDiff = Math.abs(firstPredictedPrice - lastActualPrice);
      const accuracy = Math.max(0, 100 - (priceDiff / lastActualPrice * 100));
      
      setPrediction({
        symbol,
        historicalData: data,
        predictions,
        accuracy: accuracy.toFixed(2),
        lastActualPrice: lastActualPrice.toFixed(2)
      });
      
      tf.dispose([xs, ys, xsNorm, ysNorm, model]);
    } catch (err) {
      console.error('Error in prediction:', err);
      setError('Error making prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!symbol) return;
    
    const data = await fetchHistoricalData(symbol);
    if (!data) return;
    
    const timeDiff = endDate.getTime() - startDate.getTime();
    const predictionDays = Math.ceil(timeDiff / (1000 * 60 * 60 * 24));
    
    await trainAndPredict(data, predictionDays);
  };

  return (
    <Box sx={{ maxWidth: '1000px', mx: 'auto', my: 8 }}>
      <Typography variant="h4" component="h2" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
        Stock Price Predictor (Linear Regression)
      </Typography>
      
      <FormPaper elevation={3}>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Stock Symbol"
                variant="outlined"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="e.g. AAPL, MSFT"
                InputProps={{
                  startAdornment: <InputAdornment position="start">$</InputAdornment>,
                }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <LocalizationProvider dateAdapter={AdapterDateFns}>
                <DatePicker
                  label="Start Date"
                  value={startDate}
                  onChange={(newValue) => setStartDate(newValue)}
                  renderInput={(params) => <TextField {...params} fullWidth />}
                  maxDate={new Date()}
                />
              </LocalizationProvider>
            </Grid>
            <Grid item xs={12} md={4}>
              <LocalizationProvider dateAdapter={AdapterDateFns}>
                <DatePicker
                  label="End Date"
                  value={endDate}
                  onChange={(newValue) => setEndDate(newValue)}
                  renderInput={(params) => <TextField {...params} fullWidth />}
                  minDate={startDate}
                />
              </LocalizationProvider>
            </Grid>
          </Grid>
          
          <PredictButton
            type="submit"
            variant="contained"
            size="large"
            disabled={!symbol || isLoading}
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : null}
          >
            {isLoading ? 'Predicting...' : 'Predict Prices'}
          </PredictButton>
        </form>
        
        {error && (
          <Box sx={{ mt: 3, p: 2, bgcolor: '#fee2e2', borderRadius: '8px' }}>
            <Typography color="error">{error}</Typography>
          </Box>
        )}
        
        {prediction && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
              Prediction Results for {prediction.symbol}
            </Typography>
            
            <Box sx={{ mb: 3, p: 2, bgcolor: '#f0fdf4', borderRadius: '8px' }}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body1">Last Actual Price:</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                    ${prediction.lastActualPrice}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body1">Model Accuracy:</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
                    {prediction.accuracy}%
                  </Typography>
                </Grid>
              </Grid>
            </Box>
            
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
              Price Predictions from {startDate.toLocaleDateString()} to {endDate.toLocaleDateString()}
            </Typography>
            
            <TableContainer component={Paper} sx={{ mt: 2 }}>
              <Table size="small" aria-label="prediction table">
                <TableHead>
                  <TableRow sx={{ bgcolor: '#f3f4f6' }}>
                    <TableCell sx={{ fontWeight: 'bold' }}>Date</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Predicted Price ($)</TableCell>
                    <TableCell align="right" sx={{ fontWeight: 'bold' }}>Change (%)</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {prediction.predictions.map((row, index) => {
                    const change = index > 0 
                      ? ((row.predictedPrice - prediction.predictions[index-1].predictedPrice) / 
                         prediction.predictions[index-1].predictedPrice * 100).toFixed(2)
                      : ((row.predictedPrice - parseFloat(prediction.lastActualPrice)) / 
                         parseFloat(prediction.lastActualPrice) * 100).toFixed(2);
                    
                    return (
                      <TableRow key={index}>
                        <TableCell>{row.date.toLocaleDateString()}</TableCell>
                        <TableCell align="right">${row.predictedPrice.toFixed(2)}</TableCell>
                        <TableCell 
                          align="right" 
                          sx={{ 
                            color: change >= 0 ? '#10b981' : '#ef4444',
                            fontWeight: 'bold'
                          }}
                        >
                          {change >= 0 ? '+' : ''}{change}%
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </FormPaper>
    </Box>
  );
};

export default PredictionForm;