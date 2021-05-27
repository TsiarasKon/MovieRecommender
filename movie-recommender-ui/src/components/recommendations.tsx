import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Card, Carousel, Skeleton, Spin, Tag } from 'antd';
import Movie from '../types';
import { chunks } from '../utils';
import '../styles/recommendations.css';

const RecommendationsComponent = ({ ratedMovies, allMovies }: { ratedMovies: Movie[], allMovies: Movie[] }) => {
  const [recommendedMovies, setRecommendedMovies] = useState<Movie[]>([]);
  const [responsesWaitingNum, setResponsesWaitingNum] = useState(0);
  const timerRef = useRef<any>(null);     // timer used to prevent multiple back to back API requests
  const timerDuration = 2000;

  const getRecommendations = useCallback((): void => {
    if (!ratedMovies.length) return;
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setResponsesWaitingNum(prev => prev + 1);
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
          setResponsesWaitingNum(prev => prev - 1);
        })
        .catch(error => {
          console.log("Got error: ", error);
          setResponsesWaitingNum(prev => prev - 1);
        });
    }, timerDuration)
  }, [ratedMovies, allMovies]);

  useEffect(() => getRecommendations(), [getRecommendations]);

  const recommendedMoviesCarouselCards = ([...chunks(recommendedMovies, 4)] as Movie[][]).map(mArr =>
    <div>
      {mArr.map(m =>
        <Card title={m.primaryTitle} bordered={true} key={m.tconst}>
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
      title={<Skeleton title paragraph={false} loading={!!responsesWaitingNum} active>{m.primaryTitle}</Skeleton>}
      bordered={true} className="movie-card"
      key={m.tconst}
    >
      <Skeleton title={false} paragraph={{ rows: 2 }} loading={!!responsesWaitingNum} active>
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
      {recommendedMoviesCards.length ?
        <div className="recommended-container">
          {recommendedMoviesCards}
        </div>
        :
        !!responsesWaitingNum && <Spin size="large" style={{ margin: '45%' }} />
      }
    </div>
  );
}

export default RecommendationsComponent;
