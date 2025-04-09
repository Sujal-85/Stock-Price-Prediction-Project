import React from 'react';
import { Box, Typography, Grid, Paper, Divider, Card, CardContent, useMediaQuery } from '@mui/material';
import { styled, useTheme } from '@mui/system';
import LSTMImage from '../assets/P1.png'; // Replace with your actual image paths
import GRUImage from '../assets/p3.png';
import LinearRegressionImage from '../assets/p2.jpg';

const MetricCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  borderRadius: '12px',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
  transition: 'transform 0.3s',
  '&:hover': {
    transform: 'translateY(-5px)',
  },
}));

const ModelCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: '12px',
  overflow: 'hidden',
  boxShadow: '0 6px 15px rgba(0, 0, 0, 0.1)',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 12px 20px rgba(0, 0, 0, 0.15)',
  },
}));

const AboutComponent = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  const metrics = [
    { 
      name: 'MSE', 
      value: '12.02', 
      description: [
        'Mean Squared Error Average squared difference between',
        'actual and predicted values'
      ] 
    },
    { 
      name: 'RMSE', 
      value: '3.47', 
      description: [
        'Root Mean Squared Error Square root of MSE In same units as target'
      ] 
    },
    { 
      name: 'MAE', 
      value: '2.55', 
      description: [
        'Mean Absolute Error Average absolute difference between actual',
        'and predicted'
      ] 
    },
    { 
      name: 'MAPE', 
      value: '1.47%', 
      description: [
        'Mean Absolute Percentage Error Percentage error between',
        'actual and predicted values'
      ] 
    },
    { 
      name: 'R²', 
      value: '0.93', 
      description: [
        'R-squared Score Proportion of variance explained by the ',
        'model (0-1)'
      ] 
    },
    { 
      name: 'Accuracy', 
      value: '98.53%', 
      description: [
        'Percentage accuracy Calculated based on',
        'MAPE score'
      ] 
    },
  ];

  const models = [
    {
      name: 'Linear Regression',
      image: LinearRegressionImage,
      description: 'A fundamental statistical approach that models the relationship between dependent and independent variables using a linear equation. Best for simpler relationships in stock data.',
      pros: ['Fast training', 'Easy to interpret', 'Works well with small datasets'],
      cons: ['Poor with non-linear patterns', 'Sensitive to outliers']
    },
    {
      name: 'LSTM (Long Short-Term Memory)',
      image: LSTMImage,
      description: 'Advanced recurrent neural network capable of learning long-term dependencies in time series data.Ideal for capturing complex patterns in stock price movements.',
      pros: ['Handles long sequences', 'Remembers important patterns', 'Good with volatility'],
      cons: ['Computationally expensive', 'Requires large datasets']
    },
    {
      name: 'GRU (Gated Recurrent Unit)',
      image: GRUImage,
      description: 'A variation of LSTM with fewer parameters but similar performance. Efficient for stock prediction with slightly less complex architecture.',
      pros: ['Faster than LSTM', 'Good with medium sequences', 'Efficient memory usage'],
      cons: ['Slightly less accurate than LSTM', 'Still complex']
    }
  ];

  return (
    <Box sx={{ py: 6, px: { xs: 2, md: 6 }, maxWidth: '1200px', mx: 'auto' }}>
      {/* Introduction Section */}
      <Typography variant="h3" component="h1" gutterBottom sx={{ 
        fontWeight: 'bold', 
        mb: 4,
        fontSize: isMobile ? '2rem' : '2.5rem',
        background: 'linear-gradient(45deg, #3f51b5 30%, #2196f3 90%)',
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent'
      }}>
        About StockSense AI
      </Typography>
      
      <Typography variant="h6" paragraph sx={{ lineHeight: 1.7, mb: 4, fontSize: isMobile ? '1rem' : '1.2rem' }}>
        StockSense AI is an advanced stock price prediction platform that leverages cutting-edge machine learning 
        algorithms to forecast market trends with exceptional accuracy. Our system analyzes historical price data, 
        trading volumes, and market indicators to generate reliable predictions for informed investment decisions.
      </Typography>

      {/* Performance Metrics */}
      <Typography variant="h4" component="h2" gutterBottom sx={{ 
        fontWeight: 'bold', 
        mt: 6, 
        mb: 4,
        fontSize: isMobile ? '1.5rem' : '2rem',
        color: 'primary.main'
      }}>
        Model Performance Metrics
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 6 }}>
        {metrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <MetricCard elevation={0}>
              <Typography variant="h4" component="div" sx={{ 
                fontWeight: 'bold',
                fontSize: isMobile ? '1.5rem' : '2rem',
                background: 'linear-gradient(45deg, #4caf50 30%, #8bc34a 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 1
              }}>
                {metric.value}
              </Typography>
              <Typography variant="h6" component="div" sx={{ fontWeight: 'medium', mb: 1, fontSize: isMobile ? '1rem' : '1.25rem' }}>
                {metric.name}
              </Typography>
              {metric.description.map((line, i) => (
                <Typography 
                  key={i} 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ lineHeight: 1.3, fontSize: isMobile ? '0.875rem' : '1rem' }}
                >
                  {line}
                </Typography>
              ))}
            </MetricCard>
          </Grid>
        ))}
      </Grid>

      {/* Models Section */}
      <Typography variant="h4" component="h2" gutterBottom sx={{ 
        fontWeight: 'bold', 
        mt: 6, 
        mb: 4,
        fontSize: isMobile ? '1.5rem' : '2rem',
        color: 'primary.main'
      }}>
        Prediction Models
      </Typography>
      
      <Grid container spacing={4} sx={{ mb: 4 }}>
        {models.map((model, index) => (
          <Grid item xs={12} md={4} key={index}>
            <ModelCard>
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: isMobile ? 'column' : (index % 2 === 0 ? 'row' : 'row-reverse'),
                  alignItems: 'center',
                  height: '100%'
                }}
              >
                <Box
                  component="img"
                  src={model.image}
                  alt={model.name}
                  sx={{
                    width: isMobile ? '100%' : '40%',
                    height: isMobile ? '200px' : '320px',
                    objectFit: 'cover',
                    flexShrink: 0
                  }}
                />
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography gutterBottom variant="h5" component="h3" sx={{ fontWeight: 'bold', fontSize: isMobile ? "1.25rem" : "1.5rem" }}>
                    {model.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" component="h3" sx={{ fontSize: isMobile ? "0.875rem" : "1rem" }}>
                    {model.description}
                  </Typography>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: 'success.main', fontSize: isMobile ? "0.875rem" : "1rem" }}>
                    Strengths:
                  </Typography>
                  <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
                    {model.pros.map((pro, i) => (
                      <li key={i} style={{ marginBottom: '4px' }}>
                        <Typography variant="body2" sx={{ fontSize: isMobile ? "0.875rem" : "1rem" }}>{pro}</Typography>
                      </li>
                    ))}
                  </ul>
                  
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: 'error.main', mt: 1, fontSize: isMobile ? "0.875rem" : "1rem" }}>
                    Limitations:
                  </Typography>
                  <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
                    {model.cons.map((con, i) => (
                      <li key={i} style={{ marginBottom: '4px' }}>
                        <Typography variant="body2" sx={{ fontSize: isMobile ? "0.875rem" : "1rem" }}>{con}</Typography>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Box>
            </ModelCard>
          </Grid>
        ))}
      </Grid>

      {/* How It Works Section */}
      <Typography variant="h4" component="h2" gutterBottom sx={{ 
        fontWeight: 'bold', 
        mt: 6, 
        mb: 4,
        fontSize: isMobile ? '1.5rem' : '2rem',
        color: 'primary.main'
      }}>
        How Our Prediction System Works
      </Typography>
      
      <Box sx={{ 
        p: isMobile ? 2 : 4, 
        bgcolor: 'background.paper', 
        borderRadius: 2, 
        boxShadow: 1,
        mb: 6
      }}>
        <ol style={{ paddingLeft: '20px' }}>
          <li style={{ marginBottom: '16px' }}>
            <Typography variant="h6" component="span" sx={{ fontWeight: 'bold', fontSize: isMobile ? '1rem' : '1.25rem' }}>Data Collection: </Typography>
            <Typography variant="body1" component="span" sx={{ fontSize: isMobile ? '0.875rem' : '1rem' }}>
              We gather 10+ years of historical stock data from reliable financial APIs
            </Typography>
          </li>
          <li style={{ marginBottom: '16px' }}>
            <Typography variant="h6" component="span" sx={{ fontWeight: 'bold', fontSize: isMobile ? '1rem' : '1.25rem' }}>Feature Engineering: </Typography>
            <Typography variant="body1" component="span" sx={{ fontSize: isMobile ? '0.875rem' : '1rem' }}>
              Our system identifies 25+ key technical indicators (RSI, MACD, Bollinger Bands, etc.)
            </Typography>
          </li>
          <li style={{ marginBottom: '16px' }}>
            <Typography variant="h6" component="span" sx={{ fontWeight: 'bold', fontSize: isMobile ? '1rem' : '1.25rem' }}>Model Training: </Typography>
            <Typography variant="body1" component="span" sx={{ fontSize: isMobile ? '0.875rem' : '1rem' }}>
              Multiple models are trained simultaneously using GPU-accelerated cloud computing
            </Typography>
          </li>
          <li style={{ marginBottom: '16px' }}>
            <Typography variant="h6" component="span" sx={{ fontWeight: 'bold', fontSize: isMobile ? '1rem' : '1.25rem' }}>Ensemble Prediction: </Typography>
            <Typography variant="body1" component="span" sx={{ fontSize: isMobile ? '0.875rem' : '1rem' }}>
              Predictions from all models are weighted based on their recent accuracy
            </Typography>
          </li>
          <li>
            <Typography variant="h6" component="span" sx={{ fontWeight: 'bold', fontSize: isMobile ? '1rem' : '1.25rem' }}>Result Delivery: </Typography>
            <Typography variant="body1" component="span" sx={{ fontSize: isMobile ? '0.875rem' : '1rem' }}>
              Final predictions are displayed with confidence intervals and trend analysis
            </Typography>
          </li>
        </ol>
      </Box>
    </Box>
  );
};

export default AboutComponent;