import React from 'react';
import { Box, Typography, Grid, Card, CardContent } from '@mui/material';
import { styled } from '@mui/system';

const FeatureCard = styled(Card)({
  height: '100%',
  borderRadius: '12px',
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
  transition: 'transform 0.3s, box-shadow 0.3s',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0 10px 15px rgba(0, 0, 0, 0.1)',
  },
});

const FeatureIcon = styled(Box)({
  fontSize: '2.5rem',
  marginBottom: '1rem',
  color: '#2563eb',
});

const FeaturesSection = () => {
  const features = [
    {
      icon: '📈',
      title: 'Advanced Algorithms',
      description: 'Our models use cutting-edge machine learning to analyze market trends.',
    },
    {
      icon: '🔮',
      title: 'Future Predictions',
      description: 'Get accurate forecasts for short-term and long-term stock performance.',
    },
    {
      icon: '📊',
      title: 'Visual Analytics',
      description: 'Interactive charts help you understand market patterns easily.',
    },
  ];

  return (
    <Box sx={{ py: 8 }}>
      <Typography variant="h4" component="h2" align="center" sx={{ mb: 6, fontWeight: 'bold' }}>
        Why Choose StockSense?
      </Typography>
      <Grid container spacing={4}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <FeatureCard>
              <CardContent sx={{ textAlign: 'center', p: 4 }}>
                <FeatureIcon>{feature.icon}</FeatureIcon>
                <Typography variant="h5" component="h3" gutterBottom sx={{ fontWeight: '600' }}>
                  {feature.title}
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </FeatureCard>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default FeaturesSection;