import React from 'react';
import { Box, Typography, Button, Container } from '@mui/material';
import { styled } from '@mui/system';

const HeroContainer = styled(Box)({
  background: 'linear-gradient(135deg, #2563eb 0%, #1e40af 100%)',
  color: 'white',
  padding: '6rem 0',
  borderRadius: '0 0 20px 20px',
  textAlign: 'center',
});

const HeroButton = styled(Button)({
  marginTop: '2rem',
  padding: '12px 32px',
  fontSize: '1.1rem',
  fontWeight: '600',
  borderRadius: '12px',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 6px 8px rgba(0, 0, 0, 0.15)',
  },
});

const HeroSection = () => {
  return (
    <HeroContainer>
      <Container maxWidth="md">
        <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          AI-Powered Stock Market Predictions
        </Typography>
        <Typography variant="h5" component="p" sx={{ opacity: 0.9, mb: 2 }}>
          Harness the power of machine learning to forecast stock prices with unprecedented accuracy
        </Typography>
        <HeroButton
          variant="contained"
          color="secondary"
          size="large"
          href="https://sujal-85-stock-price-prediction-project-app-lxl0hw.streamlit.app/"
          target="_blank" // Opens in a new tab
          rel="noopener noreferrer" // Recommended for security
>
          Try Stock Predictor
        </HeroButton>
      </Container>
    </HeroContainer>
  );
};

export default HeroSection;