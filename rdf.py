import glob
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import text2emotion as te

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
ns_wiki = Namespace('http://www.wikidata.org/entity/')


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

    return movies_df, principals_df


def review_data_loading():
    num_reviews = 20
    reviews_path = os.path.join("imdb_reviews", "aclImdb", "train")
    reviews_files = glob.glob(os.path.join(reviews_path, "unsup", "*.txt"))[:num_reviews]
    reviews_texts = []
    for r_f in reviews_files:
        with open(r_f, "r", encoding="utf8") as f:
            reviews_texts.append(f.readlines()[0])

    reviews_emotions = [te.get_emotion(t) for t in tqdm(reviews_texts)]
    reviews_tconst = []
    with open(os.path.join(reviews_path, "urls_unsup.txt"), "r") as f:
        for line in f:
            reviews_tconst.append(line.split('/')[4])
    print([reviews_tconst[int(f.split(os.path.sep)[-1].split('_')[0])] for f in reviews_files])
    reviews_df = pd.DataFrame({
        "tconst": [reviews_tconst[int(f.split(os.path.sep)[-1].split('_')[0])] for f in reviews_files],
        # "text": reviews_texts,
        "Happy": [e["Happy"] for e in reviews_emotions],
        "Angry": [e["Angry"] for e in reviews_emotions],
        "Surprise": [e["Surprise"] for e in reviews_emotions],
        "Sad": [e["Sad"] for e in reviews_emotions],
        "Fear": [e["Fear"] for e in reviews_emotions],
    }).set_index("tconst")
    print(reviews_df)
    print(reviews_df.groupby(["tconst"]).mean())


def build_and_save_rdf(save=True, limit=None, prune=True):
    movies_df, principals_df = data_loading()

    g = Graph()

    # Get extra data from wikidata
    wikidata_df = get_all_from_wikidata(movies_df.index.tolist())
    print(wikidata_df)

    for imdb_id, data in tqdm(movies_df.iterrows(), total=movies_df.shape[0] if limit is None else min(movies_df.shape[0], limit), desc='Building rdf graph'):
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
        # votes = Literal(str(data['numVotes']), datatype=XSD.integer)
        # has_votes = URIRef(ns_predicates + 'hasVotes')
        # g.add((movie, has_votes, votes))

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
            if pred is not None:
                g.add((movie, pred, URIRef(ns_principals + artist)))

        # TODO: are info for artists useful for our task? Should we add such triples?

        # create rdf graph from wikidata  TODO: use codes instead of names?
        if imdb_id in wikidata_df.index:
            has_series = URIRef(ns_predicates + 'hasSeries')
            series = wikidata_df.loc[imdb_id]['series'].split('; ')
            for s in series:
                if len(s) > 0 and s.startswith('http'): g.add((movie, has_series, URIRef(s)))

            has_distributor = URIRef(ns_predicates + 'hasDistributor')
            distributors = wikidata_df.loc[imdb_id]['distributors'].split('; ')
            for d in distributors:
                if len(d) > 0 and d.startswith('http'): g.add((movie, has_distributor, URIRef(d)))

            has_subject = URIRef(ns_predicates + 'hasSubject')
            subjects = wikidata_df.loc[imdb_id]['subject'].split('; ')
            for s in subjects:
                if len(s) > 0 and s.startswith('http'): g.add((movie, has_subject, URIRef(s)))

    # Prune graph of unused stuff
    if prune:
        prune_rdf_graph(g)

    # Save graph
    if save:
        print('Saving graph...')
        g.serialize(destination='movies.nt', format='nt')

    return g


def prune_rdf_graph(g: Graph, actor_least_movies=3, director_least_movies=5):
    print('Deleting unused actors...')
    g.update(
        """ DELETE { ?m pred:hasActor ?a }
            WHERE {
                ?m pred:hasActor ?a .
                FILTER EXISTS {
                    SELECT DISTINCT ?a
                    WHERE {
                        ?movie pred:hasActor ?a .
                    }
                    GROUP BY ?a
                    HAVING (COUNT(?movie) < """ + str(actor_least_movies) + """)
                }
            }
        """, initNs={'movies': ns_movies,
                     'genres': ns_genres,
                     'pred': ns_predicates,
                     'principals': ns_principals})

    print('Deleting unused directors...')
    g.update(
        """ DELETE { ?m ?p ?d }
            WHERE {
                ?m pred:hasDirector ?d .
                FILTER EXISTS {
                    SELECT DISTINCT ?d
                    WHERE {
                        ?movie pred:hasDirector ?d .
                    }
                    GROUP BY ?d
                    HAVING (COUNT(?movie) < """ + str(director_least_movies) + """)
                    }
                }
            """, initNs={'movies': ns_movies,
                         'genres': ns_genres,
                         'pred': ns_predicates,
                         'principals': ns_principals})


def load_rdf():
    g = Graph()
    return g.parse('movies.nt', format='nt')


if __name__ == '__main__':
    only_load = False    # load or build from scratch?
    prune_loaded = False

    if only_load:
        print('Loading rdf...')
        g = load_rdf()
        print('done')

        if prune_loaded:
            prune_rdf_graph(g)
            print('Saving graph...')
            g.serialize(destination='movies.nt', format='nt')
    else:
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
