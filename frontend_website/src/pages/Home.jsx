import React from 'react';
import Header from '../components/Navbar';
import HeroSection from '../components/HeroSection';
import FeaturesSection from '../components/FeaturesSection';
import AboutComponent from '../components/AboutComponent';
import { Box } from '@mui/material';

const Home = () => {
  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <Header />
      <HeroSection />
      <FeaturesSection />
      <AboutComponent />
      {/* Add more sections as needed */}
    </Box>
  );
};

export default Home;