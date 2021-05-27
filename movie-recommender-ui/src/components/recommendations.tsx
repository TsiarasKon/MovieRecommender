import React, { useCallback, useEffect, useState } from 'react';
import { Card, Carousel, Skeleton, Tag } from 'antd';
import Movie from '../types';
import { chunks } from '../utils';
import '../styles/recommendations.css';

const RecommendationsComponent = ({ ratedMovies, allMovies }: { ratedMovies: Movie[], allMovies: Movie[] }) => {
  const [recommendedMovies, setRecommendedMovies] = useState<Movie[]>([]);
  const [loadingFlag, setLoadingFlag] = useState(false);

  const getRecommendations = useCallback((): void => {
    if (!ratedMovies.length) return;
    setLoadingFlag(true);
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        movieRatings: ratedMovies.map(m => ({ tconst: m.tconst, userRating: m.userRating })),
        recommendationsNum: 10
      })
    };
    console.log("Sending request with body: ", requestOptions.body)
    fetch('/recommend', requestOptions)
      .then(response => response.json())
      .then(data => {
        console.log("Got response: ", data);
        const recommendedTconstArr = data.map((el: any) => el.tconst);
        setRecommendedMovies(allMovies.filter(m => recommendedTconstArr.includes(m.tconst)));
        setLoadingFlag(false);
      })
  }, [ratedMovies, allMovies]);

  useEffect(() => getRecommendations(), [getRecommendations]);

  const recommendedMoviesCarouselCards = ([...chunks(recommendedMovies, 4)] as Movie[][]).map(mArr =>
    <div>
      {mArr.map(m =>
        <Card title={m.primaryTitle} bordered={true}>
          Year: <strong>{m.startYear}</strong>
          <br />
          Rating: <strong>{m.averageRating}</strong> <i>({m.numVotes} votes)</i>
          <br />
          Genres: {m.genres && (m.genres as unknown as string).split(',').map((g: string) =>
            <Tag key={g} style={{ margin: 2 }}>
              {g}
            </Tag>)
          }
        </Card>
      )}
    </div>
  );

  const recommendedMoviesCards = recommendedMovies.map(m =>
    <Card
      title={<Skeleton title paragraph={false} loading={loadingFlag} active>{m.primaryTitle}</Skeleton>}
      bordered={true} className="movie-card"
    >
      <Skeleton title={false} paragraph={{ rows: 2 }} loading={loadingFlag} active>
        Year: <strong>{m.startYear}</strong>
        <br />
      Rating: <strong>{m.averageRating}</strong> <i>({m.numVotes} votes)</i>
        <br />
      Genres: {m.genres && (m.genres as unknown as string).split(',').map((g: string) =>
          <Tag key={g} style={{ margin: 2 }}>
            {g}
          </Tag>)
        }
      </Skeleton>
    </Card>
  );

  return (
    <div className="recommended-section">
      <h2 className="text-center">Recommendations</h2>
      {/* <Carousel dotPosition="right" infinite={false}>
        {recommendedMoviesCards}
      </Carousel> */}
      <div className="recommended-container">
        {recommendedMoviesCards}
      </div>
    </div>
  );
}

export default RecommendationsComponent;
