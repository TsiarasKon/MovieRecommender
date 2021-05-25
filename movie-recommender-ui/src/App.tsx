import MovieTableComponent from './movieTable';
import RecommendationsComponent from './recommendations';

function App() {
  return (
    <div style={{width: '90%', margin: '0 auto 0 auto',}}>
      <div style={{textAlign: 'center'}}><h1>Movie Recommender</h1></div>
      <MovieTableComponent />
    </div>
  );
}

export default App;
