interface Principal {
    nconst: string,
    name: string,
    profession: string[],       // TODO: enum?
    knownFor: Movie[],
    birthYear?: number,
    deathYear?: number
}

interface Movie {
    tconst: string,
    primaryTitle: string,
    genres: string[],
    startYear: number,
    runtimeMinutes?: number,
    isAdult?: boolean,
    averageRating: number,
    numVotes: number
    director?: Principal,
    userRating?: number
}

export default Movie;