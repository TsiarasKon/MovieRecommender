import { Row, Col, Card, Carousel } from 'antd';
import React, { useEffect, useState } from 'react';
import MovieTableComponent from './movieTable';
import Movie from './types';
import { chunks } from './utils';


const RecommendationsComponent = ({ ratedMovies, allMovies }: { ratedMovies: Movie[], allMovies: Movie[] }) => {

  const [recommendedMovies, setRecommendedMovies] = useState([]);
  const carouselMoviesArray = [...chunks(recommendedMovies, 4)] as Movie[][]
  console.log(carouselMoviesArray)

  const getRecommendations = (): void => {
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        movieRatings: ratedMovies,
        recommendationsNum: 10
      })
    };
    console.log("Sending request with body: ", requestOptions.body)
    fetch('/recommend', requestOptions)
      .then(response => response.json())
      .then(data => console.log(data))
      // .then(data => setRecommendedMovies())
  }

  // useEffect(() => getRecommendations(), []);

  // const ratedMovies = carouselMoviesArray.map(mArr =>
  //   <div>
  //     {mArr.map(m =>
  //       <Card title={m.primaryTitle} bordered={true} style={{ height: '20%', paddingRight: 20 }}>
  //         {m.tconst}
  //       </Card>
  //     )}
  //   </div>
  // );

  return (
    <div className="site-card-wrapper" style={{ background: "lightgray", padding: 20, height: 1000 }}>
      <h2 style={{ textAlign: "center" }}>Recommendations</h2>
      <Carousel dotPosition="right">
        {ratedMovies}
      </Carousel>
      {/* {getRecommendations().m =>
        <Card title={m.primaryTitle} bordered={true} style={{ height: '20%', paddingRight: 20 }}>
          {m.tconst}
        </Card>
      } */}
    </div>
  );
}

export default RecommendationsComponent;
