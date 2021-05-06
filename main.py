import pandas as pd
import numpy as np
from tqdm import tqdm

from rdflib import Graph, Literal, URIRef, Namespace, XSD

imdb_data_folder = 'imdb_data/'
name_basics_file = 'name.basics.tsv'
tconst_files = [
    'title.basics.tsv',
    # 'title.crew.tsv',
    # 'title.episode.tsv',
    # 'title.principals.tsv',
    'title.ratings.tsv'
]

movielens_data_folder = 'movielens_data/'

pd.set_option('display.max_columns', None)


def data_loading():
    # load each file with tconst (title id) as index
    print('Loading IMDB data')
    all_dfs = []
    for file in tconst_files:
        df = pd.read_csv(imdb_data_folder + file, index_col='tconst',  # usecols=[], dtype={'col': 'UInt32'}
                         sep='\t', encoding='utf-8',
                         keep_default_na=False, na_values=['\\N'])
        all_dfs.append(df)

    # combine all into one big fat DataFrame
    print('concatenating...')
    full_df = pd.concat(all_dfs, axis=1)
    full_df = full_df[(full_df['titleType'].isin(['tvMovie', 'movie'])) &
                      (full_df['startYear'] >= 1960) & (full_df['startYear'] <= 2020)]
    print('done')

    # fix NA and types afterwards as it is not supported for read_csv
    full_df['numVotes'] = full_df['numVotes'].fillna(0).astype(np.uint16)
    full_df['isAdult'] = full_df['isAdult'].astype(np.bool)
    full_df['startYear'] = full_df['startYear'].fillna(0).astype(np.uint16)
    full_df['endYear'] = full_df['endYear'].fillna(0).astype(np.uint16)
    full_df['genres'] = full_df['genres'].fillna('').astype(np.str)

    # filtering
    full_df = full_df[(full_df['numVotes'] >= 500) &
                      (~(full_df['genres'].str.contains('Short', regex=False, na=False))) &
                      (full_df['genres'].str != '')]

    print(full_df)
    return full_df


# def build_rdf():
if __name__ == '__main__':
    df = data_loading()

    ns_movies = Namespace('https://www.imdb.com/title/')
    ns_predicates = Namespace('http://example.org/props/')
    ns_xsd = Namespace('http://www.w3.org/2001/XMLSchema#')

    g = Graph()

    for imdb_id, data in tqdm(df.iterrows(), total=df.shape[0]):
        movie = URIRef(ns_movies + imdb_id)

        # add genres
        has_genre = URIRef(ns_predicates + 'hasGenre')
        genres = data['genres'].split(',')
        genres = [g.replace(' ', '') for g in genres]
        for genre_str in genres:
            genre = URIRef(ns_movies + genre_str)
            g.add((movie, has_genre, genre))

        # add years
        year = Literal(str(data['startYear']), datatype=XSD.integer)
        has_year = URIRef(ns_predicates + 'hasYear')
        g.add((movie, has_year, year))

    qres = g.query(
        """SELECT DISTINCT ?x
           WHERE {
              ?x pred:hasGenre ns:Drama .
              ?x pred:hasYear 2003 .
           }""", initNs={'ns': ns_movies, 'pred': ns_predicates})

    print(f'Results: {len(qres)}')
    for row in qres:
        # print("%s has genre 'Drama'" % row)
        print(row)
