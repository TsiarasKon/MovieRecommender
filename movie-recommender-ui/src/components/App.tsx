import React, { useState } from 'react';
import { Row, Col } from 'antd';
import MovieTableComponent from './movieTable';
import RecommendationsComponent from './recommendations';
import Movie from '../types';
import { loadJsonData } from '../utils';
import MoviesData from "../data/movies.json";

function App() {
  const AllMovies: Movie[] = JSON.parse(loadJsonData(MoviesData));
  const [movies, setMovies] = useState(AllMovies);

  return (
    <div>
        <div className={"header"}>
            <h1 className="text-center">Movie Recommender</h1>
        </div>
        <div className="app-container">
            <Row justify="space-between">
            <Col span={17}>
              <MovieTableComponent allMovies={movies} setParentMovies={(newMovies: Movie[]) => setMovies(newMovies)} />
            </Col>
            <Col span={6}>
              <RecommendationsComponent ratedMovies={movies.filter(m => !!m.userRating)} allMovies={AllMovies} />
            </Col>
            </Row>
        </div>
    </div>
  );
}

export default App;
