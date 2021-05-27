import MovieTableComponent from './movieTable';

function App() {
  return (
    <div style={{ width: '90%', margin: '0 auto 0 auto' }}>
      <div style={{ textAlign: 'center', paddingTop: 10 }}><h1>Movie Recommender</h1></div>
      <MovieTableComponent />
    </div>
  );
}

export default App;
