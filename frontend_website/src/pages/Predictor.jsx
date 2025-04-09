import React from 'react';
import Header from '../components/Navbar';
import StockChart from '../components/StockChart';
import PredictionForm from '../components/PredictionForm';
import { Box, Container } from '@mui/material';

const Predictor = () => {
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <Header />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <PredictionForm />
        <Box sx={{ mt: 6 }}>
          {/* <StockChart /> */}
        </Box>
      </Container>
    </Box>
  );
};

export default Predictor;