import React from 'react';
import { Box, Container, Grid, Typography, Link, Divider } from '@mui/material';
import { styled } from '@mui/system';

const FooterContainer = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.secondary.main,
  color: 'white',
  padding: theme.spacing(6, 0),
  marginTop: 'auto',
}));

const FooterLink = styled(Link)(({ theme }) => ({
  color: 'rgba(255, 255, 255, 0.7)',
  display: 'block',
  marginBottom: theme.spacing(1),
  '&:hover': {
    color: 'white',
    textDecoration: 'none',
  },
}));

const SocialIcon = styled(Box)(({ theme }) => ({
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 40,
  height: 40,
  borderRadius: '50%',
  backgroundColor: 'rgba(255, 255, 255, 0.1)',
  marginRight: theme.spacing(1),
  color: 'white',
  transition: 'all 0.3s ease',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    transform: 'translateY(-2px)',
  },
}));

const Footer = () => {
  return (
    <FooterContainer component="footer">
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', color: 'white' }}>
              StockSense
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 2 }}>
              Harnessing AI to predict stock market trends with unprecedented accuracy.
            </Typography>
            <Box sx={{ display: 'flex', mt: 2 }}>
              <SocialIcon component="a" href="#">
                <i className="fab fa-twitter"></i>
              </SocialIcon>
              <SocialIcon component="a" href="#">
                <i className="fab fa-linkedin-in"></i>
              </SocialIcon>
              <SocialIcon component="a" href="#">
                <i className="fab fa-github"></i>
              </SocialIcon>
            </Box>
          </Grid>
          <Grid item xs={6} md={2}>
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'white' }}>
              Product
            </Typography>
            <FooterLink href="#" variant="body2">
              Features
            </FooterLink>
            <FooterLink href="#" variant="body2">
              Pricing
            </FooterLink>
            <FooterLink href="#" variant="body2">
              API
            </FooterLink>
            <FooterLink href="#" variant="body2">
              Documentation
            </FooterLink>
          </Grid>
          <Grid item xs={6} md={2}>
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'white' }}>
              Company
            </Typography>
            <FooterLink href="#" variant="body2">
              About
            </FooterLink>
            <FooterLink href="#" variant="body2">
              Blog
            </FooterLink>
            <FooterLink href="#" variant="body2">
              Careers
            </FooterLink>
            <FooterLink href="#" variant="body2">
              Contact
            </FooterLink>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold', color: 'white' }}>
              Subscribe to our newsletter
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 2 }}>
              Get the latest updates and stock predictions.
            </Typography>
            <Box component="form" sx={{ display: 'flex' }}>
              <input
                type="email"
                placeholder="Your email"
                style={{
                  flexGrow: 1,
                  padding: '10px 16px',
                  border: 'none',
                  borderRadius: '8px 0 0 8px',
                  fontSize: '0.875rem',
                }}
              />
              <button
                type="submit"
                style={{
                  backgroundColor: '#2563eb',
                  color: 'white',
                  border: 'none',
                  padding: '0 16px',
                  borderRadius: '0 8px 8px 0',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
              >
                Subscribe
              </button>
            </Box>
          </Grid>
        </Grid>
        <Divider sx={{ my: 4, backgroundColor: 'rgba(255, 255, 255, 0.1)' }} />
        <Typography variant="body2" align="center" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
          © {new Date().getFullYear()} StockSense. All rights reserved.
        </Typography>
      </Container>
    </FooterContainer>
  );
};

export default Footer;