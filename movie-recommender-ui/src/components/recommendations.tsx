import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Badge, Card, Skeleton, Spin, Tag } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';
import Movie from '../types';
import '../styles/recommendations.css';

const RecommendationsComponent = ({ ratedMovies, allMovies }: { ratedMovies: Movie[], allMovies: Movie[] }) => {
  const [recommendedMovies, setRecommendedMovies] = useState<Movie[]>([]);
  const [responsesWaitingNum, setResponsesWaitingNum] = useState(0);
  const [serverErrorFlag, setServerErrorFlag] = useState(false);
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
          setServerErrorFlag(false);
          const recommendedTconstArr = data.map((el: any) => el.tconst);
          setRecommendedMovies(allMovies.filter(m => recommendedTconstArr.includes(m.tconst)));
          setResponsesWaitingNum(prev => prev - 1);
        })
        .catch(error => {
          console.log("Got error: ", error);
          setServerErrorFlag(true);
          setResponsesWaitingNum(prev => prev - 1);
        });
    }, timerDuration)
  }, [ratedMovies, allMovies]);

  useEffect(() => ratedMovies.length ? getRecommendations() : setRecommendedMovies([]), [getRecommendations, ratedMovies]);

  const recommendedMoviesCards = recommendedMovies.map((m, i) =>
    <a href={'https://www.imdb.com/title/' + m.tconst} style={{ display: 'block' }} target={"_blank"} rel="noreferrer">
      <Card
        title={<Skeleton title paragraph={false} loading={!!responsesWaitingNum} active>
          <h4 style={{ marginBottom: 0 }}><Badge count={i + 1} className="card-badge" /> {m.primaryTitle}</h4>
        </Skeleton>
        }
        bordered={true} className="movie-card" key={m.tconst} hoverable={!responsesWaitingNum}
      >
        <Skeleton title={false} paragraph={{ rows: 2 }} loading={!!responsesWaitingNum} active>
          Year: &nbsp; <strong>{m.startYear}</strong>
          <br />
        Rating: &nbsp; <strong>{m.averageRating}</strong> &nbsp; <span style={{ opacity: 0.75 }}>({m.numVotes} votes)</span>
          <br />
        Genres: {m.genres && (m.genres as unknown as string).split(',').map((g: string) =>
            <Tag key={g} style={{ margin: 2 }}>
              {g}
            </Tag>)
          }
        </Skeleton>
      </Card>
    </a>
  );

  return (
    <div className="recommended-section">
      <h2 className="text-center">
        Recommendations
        <hr />
      </h2>
      {serverErrorFlag ?
        <h5 className="text-center">Failed to communicate with the server. Please refresh the page.</h5>
        :
        (recommendedMoviesCards.length ?
          <div className="recommended-container">
            {recommendedMoviesCards}
          </div>
          :
          !!responsesWaitingNum && <Spin size="large" style={{ margin: '45%' }} indicator={<LoadingOutlined style={{ fontSize: 48, color: "gold" }} spin />} />
        )
      }
    </div>
  );
}

export default RecommendationsComponent;
