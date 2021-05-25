import React, { useState } from 'react';
import MovieTableComponent from './movieTable';
import Movie from './types';


function RecommendationsComponent({movies}: any) {
  
  console.log(movies);

  const ratedMovies = (movies as Movie[]).map((movie: Movie) => 
    <li>Movie: {movie.primaryTitle} tconst: {movie.tconst}</li>
  );

  return (
    <div>
      Recommendations Component
      <ul>
        {ratedMovies}
      </ul>
    </div>
  );
}

export default RecommendationsComponent;
