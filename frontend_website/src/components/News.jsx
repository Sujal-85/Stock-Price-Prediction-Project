import React from "react";
import { useState, useEffect } from 'react';
import '../styles/styles.css';

const News = () => {
  const [newsItems, setNewsItems] = useState([]);


  useEffect(() => {
    fetchNews('Stock Market'); // Default fetch when the component loads
  }, []);

  const fetchNews = async (query) => {
    // Fetching news based on category or search
    const response = await fetch(`https://newsapi.org/v2/everything?q=${query}&apiKey=eb1c263acfae4a35b1df5dfbbffdabac`);
    const data = await response.json();
    setNewsItems(data.articles);
  };

 
  return (
    <>
      <main>
      <br/><br/><br/><br/>
        <div className="cards-container container flex mt-20" style={{ fontSize: "15px" }}>
          {newsItems.map((item, index) => (
            <div className="card" key={index}>
              <div className="card-header">
                <img src={item.urlToImage || 'https://png.pngtree.com/background/20220726/original/pngtree-404-error-page-not-found-picture-image_1822651.jpg'} alt="news" />
              </div>
              <div className="card-content">
                {/* News Title as Clickable Link */}
                <h3>
                  <a href={item.url} target="_blank" rel="noopener noreferrer">
                    {item.title}
                  </a>
                </h3>
                <h6 className="news-source">{item.source.name} {new Date(item.publishedAt).toLocaleDateString()}</h6>
                <p>{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </main>
    </>
  );
};

export default News;
