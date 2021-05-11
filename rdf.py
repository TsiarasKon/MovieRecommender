import pandas as pd
import numpy as np
from tqdm import tqdm

from rdflib import Graph, Literal, URIRef, Namespace, XSD

from wikidata_service import get_all_from_wikidata

imdb_data_folder = 'imdb_data/'
name_basics_file = 'name.basics.tsv'
tconst_files = [
    ('title.basics.tsv', None),
    # 'title.crew.tsv',
    # 'title.episode.tsv',
    # ('title.principals.tsv', ['tconst', 'nconst', 'category']),
    ('title.ratings.tsv', None)
]
movielens_data_folder = 'movielens_data/'

pd.set_option('display.max_columns', None)


# NAMESPACES
ns_movies = Namespace('https://www.imdb.com/title/')
ns_genres = Namespace('https://www.imdb.com/search/title/?genres=')
ns_principals = Namespace('https://www.imdb.com/name/')
ns_predicates = Namespace('http://example.org/props/')


def data_loading():
    # load each file with tconst (title id) as index
    print('Loading IMDB data')
    all_dfs = []
    for file, usecols in tconst_files:
        df = pd.read_csv(imdb_data_folder + file, index_col='tconst',  usecols=usecols,
                         sep='\t', encoding='utf-8',
                         keep_default_na=False, na_values=['\\N'])
        all_dfs.append(df)

    # combine all into one big fat DataFrame
    print('concatenating...')
    movies_df = pd.concat(all_dfs, axis=1)
    movies_df = movies_df[(movies_df['titleType'].isin(['tvMovie', 'movie'])) &
                          (movies_df['startYear'] >= 1975) & (movies_df['startYear'] <= 2020)]
    print('done')

    # fix NA and types afterwards as it is not supported for read_csv
    movies_df['numVotes'] = movies_df['numVotes'].fillna(0).astype(np.uint16)
    movies_df['isAdult'] = movies_df['isAdult'].astype(np.bool)
    movies_df['startYear'] = movies_df['startYear'].fillna(0).astype(np.uint16)
    movies_df['endYear'] = movies_df['endYear'].fillna(0).astype(np.uint16)
    movies_df['genres'] = movies_df['genres'].fillna('').astype(np.str)

    # filtering
    movies_df = movies_df[(movies_df['numVotes'] >= 500) &
                          (~(movies_df['genres'].str.contains('Short', regex=False, na=False))) &
                          (movies_df['genres'].str != '')]

    print('Loading edges')
    principals_df = pd.read_csv(imdb_data_folder + 'title.principals.tsv',
                             sep='\t',
                             encoding='utf-8',
                             keep_default_na=False,
                             na_values=['\\N'],
                             index_col='tconst',
                             usecols=['tconst', 'nconst', 'category'])
    principals_df = principals_df[principals_df.index.isin(movies_df.index)]
    principals_df = principals_df[principals_df['category'].isin(['actor', 'actress', 'writer', 'director', 'composer'])]  # TODO: change if more roles

    print(movies_df)
    print(principals_df)

    # TODO: do we need these?
    # print('loading people info')
    # people = pd.read_csv(imdb_data_folder + 'name.basics.tsv',
    #                      sep='\t',
    #                      encoding='utf-8',
    #                      keep_default_na=False,
    #                      na_values=['\\N'],
    #                      usecols=['nconst', 'primaryName'])  # TODO: need more?
    # filtered = people[people['nconst'].isin(movies_df.get_level_values('nconst'))]
    # people_df = pd.DataFrame(index=filtered['nconst'], data={'name': list(filtered['primaryName'])})

    return movies_df, principals_df


def build_and_save_rdf(save=True, limit=None):
    movies_df, principals_df = data_loading()

    g = Graph()

    for imdb_id, data in tqdm(movies_df.iterrows(), total=movies_df.shape[0] if limit is None else min(movies_df.shape[0], limit)):
        movie = URIRef(ns_movies + imdb_id)

        # find artists involved
        try:
            artists = principals_df.loc[imdb_id]
        except KeyError:
            # if there are not any noteworthy artists involved then it's probably not a very good film and should be ignored
            continue

        # add genres
        has_genre = URIRef(ns_predicates + 'hasGenre')
        genres = data['genres'].split(',')
        genres = [g.replace(' ', '') for g in genres]
        for genre_str in genres:
            genre = URIRef(ns_genres + genre_str)
            g.add((movie, has_genre, genre))

        # add year
        year = Literal(str(data['startYear']), datatype=XSD.integer)
        has_year = URIRef(ns_predicates + 'hasYear')
        g.add((movie, has_year, year))

        # add IMDb rating and votes
        rating = Literal(str(data['averageRating']), datatype=XSD.float)
        has_rating = URIRef(ns_predicates + 'hasRating')
        g.add((movie, has_rating, rating))
        votes = Literal(str(data['numVotes']), datatype=XSD.integer)
        has_votes = URIRef(ns_predicates + 'hasVotes')
        g.add((movie, has_votes, votes))

        # add artists such as actors, directors, etc
        for artist, category in zip(artists['nconst'], artists['category']):
            pred = None
            if category == 'actor' or category == 'actress':
                pred = URIRef(ns_predicates + 'hasActor')
            elif category == 'director':
                pred = URIRef(ns_predicates + 'hasDirector')
            elif category == 'writer':
                pred = URIRef(ns_predicates + 'hasWriter')
            elif category == 'composer':
                pred = URIRef(ns_predicates + 'hasComposer')
            # TODO: editor and cinematographer?
            if pred is not None:
                g.add((movie, pred, URIRef(ns_principals + artist)))

        # TODO: are info for artists useful for our task? Should we add such triples?

    # TODO: create rdf graph from wikidata
    # wikidata_df = get_all_from_wikidata(movies_df.index.tolist())
    # print(wikidata_df)

    # Save graph
    if save:
        print('Saving graph...')
        g.serialize(destination='movies.nt', format='nt')

    return g


def load_rdf():
    g = Graph()
    return g.parse('movies.nt', format='nt')


if __name__ == '__main__':
    only_load = True    # load or build from scratch?

    if only_load:
        print('Loading rdf...')
        g = load_rdf()
        print('done')
    else:
        # g = build_and_save_rdf(save=False, limit=10)
        g = build_and_save_rdf()

    # Test Query
    qres = g.query(
        """SELECT DISTINCT ?x ?year ?director ?genre
           WHERE {              ?x pred:hasGenre ?genre .
              ?x pred:hasYear ?year .
              ?x pred:hasDirector ?director .
              ?x pred:hasActor principals:nm0000120 .
              FILTER(?year >= 2010)
           }""", initNs={'movies': ns_movies,
                         'genres': ns_genres,
                         'pred': ns_predicates,
                         'principals': ns_principals})

    print(f'Results: {len(qres)}')
    for row in qres:
        print(row)
