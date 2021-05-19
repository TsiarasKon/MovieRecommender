import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from rdflib import Graph
from tqdm import tqdm
import warnings
from bidict import bidict

from rdf import ns_movies, ns_genres, ns_predicates, ns_principals, load_rdf
from utils import save_dict, load_dict, create_user_rating_dict, load_user_ratings


movielens_data_folder = 'movielens_data/'
max_year = 2020
min_year = 1975


def extract_binary_features(actual_values: set, ordered_possible_values: list) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform([actual_values])
    return binary_format


def build_items_feature_vetors(rdf: Graph, save=True) -> (dict, np.array):   # list parallel to 2d np array's columns --> the title ID for the row
    """ Builds a feature vector for each movie """
    # get all possible categorical features
    print('Looking up genres...')
    all_genres = rdf.query(
        """ SELECT DISTINCT ?genre
            WHERE {
                ?movie pred:hasGenre ?genre . 
            }""", initNs={'pred': ns_predicates})
    all_genres = sorted([str(g['genre']) for g in all_genres if len(str(g['genre']).split('=')[-1]) > 0])
    print('Found', len(all_genres), 'genres.')

    print('Looking up actors...')
    LEAST_MOVIES = 3   # Ignore insignificant actors
    all_actors = rdf.query(
        """ SELECT DISTINCT ?actor
            WHERE {
                ?movie pred:hasActor ?actor . 
            } 
            GROUP BY ?actor 
            HAVING (COUNT(?movie) >= """ + str(LEAST_MOVIES) + ')',
        initNs={'pred': ns_predicates})
    all_actors = sorted([str(a['actor']) for a in all_actors])
    # Note: keep just the id with: actors = sorted([str(a['actor']).split('/')[-1] for a in actors])
    print('Found', len(all_actors), 'actors with at least', LEAST_MOVIES, 'movies made.')

    print('Looking up directors...')
    LEAST_MOVIES2 = 6
    all_directors = rdf.query(
        """ SELECT DISTINCT ?director
            WHERE {
                ?movie pred:hasDirector ?director . 
            }
            GROUP BY ?director
            HAVING (COUNT(?movie) >= """ + str(LEAST_MOVIES2) + ')',
        initNs={'pred': ns_predicates})
    all_directors = sorted([str(d['director']) for d in all_directors])
    print('Found', len(all_directors), 'directors with at least', LEAST_MOVIES2, 'movies directed.')

    # Query all movies on rdf and their associated features
    print('Querying movie features...')
    movies = rdf.query(
        """SELECT DISTINCT ?movie ?year ?rating
              (group_concat(distinct ?genre; separator=",") as ?genres)
              (group_concat(distinct ?actor; separator=",") as ?actors)
              (group_concat(distinct ?director; separator=",") as ?directors)
           WHERE { 
              ?movie pred:hasYear ?year .
              ?movie pred:hasRating ?rating .
              ?movie pred:hasGenre ?genre .
              ?movie pred:hasDirector ?director .
              ?movie pred:hasActor ?actor .
           } 
           GROUP BY ?movie ?year ?rating""",
        initNs={'movies': ns_movies,
                'genres': ns_genres,
                'pred': ns_predicates,
                'principals': ns_principals})
    print('Done.')

    NUM_FEATURES = 2 + len(all_genres) + len(all_actors) + len(all_directors)  # TODO
    movie_pos: bidict = bidict({})

    print('Allocating memory...')
    item_features = np.zeros((len(movies), NUM_FEATURES), dtype=np.float32)   # takes too long :(

    print('Creating item feature matrix...')
    for i, movie_data in tqdm(enumerate(movies), total=len(movies)):
        # add movie_id to parallel vector
        movie_pos[movie_data['movie'].split('/')[-1]] = i   # dict with position in item_features

        # get numerical features
        rating = float(movie_data['rating']) / 10
        year = float(int(movie_data['year']) - min_year) / (max_year - min_year)     # min-max scaling

        # Convert all categorical to binary format
        genres = set(movie_data['genres'].split(','))
        actors = set(movie_data['actors'].split(','))
        directors = set(movie_data['directors'].split(','))
        with warnings.catch_warnings():
            # hide user warnings about ignoredmissing values, ignoring these values is the desired behaviour
            warnings.simplefilter("ignore")
            genres_feat = extract_binary_features(genres, all_genres)
            actors_feat = extract_binary_features(actors, all_actors)
            directors_feat = extract_binary_features(directors, all_directors)

        # TODO (EXTRA): Add a different weight to each feature with which to experiment "balancing"?  How do we change this afterwards? Must also store it?
        pass

        # TODO: Concat all of them into one big feature vector
        item_features[i, 0] = rating
        item_features[i, 1] = year
        item_features[i, 2: 2 + len(all_genres)] = genres_feat
        item_features[i, 2 + len(all_genres): 2 + len(all_genres) + len(all_actors)] = actors_feat
        item_features[i, 2 + len(all_genres) + len(all_actors):] = directors_feat

    # save the result
    if save:
        print('Saving...')
        np.save('item_features.npy', item_features)
        save_dict(movie_pos, 'movie_pos')

    print('Done.')

    return movie_pos, item_features


def build_user_feature_vector(user_ratings: dict, movie_pos: bidict, item_features: np.array, temperature=1.0):
    """ Takes as input a user's ratings on IMDb titles and construct its user vector """
    # Note: higher temperature helps have higher values (closer to 1 and -1) when too many features are needed to even come close
    avg_rating = 2.5    # TODO: use this as average or a user average? Or maybe the minimum of the two?
    user_vector = np.zeros(item_features.shape[1], dtype=np.float64)
    missing = 0
    count = 0
    for movie_id, rating in user_ratings.items():
        try:
            pos = movie_pos[movie_id]
            # TODO: add weights?
            # take the normal average for numerical features
            user_vector[:2] += item_features[pos, :2]
            # use weights based on rating for categorical features
            user_vector[2:] += temperature * ((rating - avg_rating) / avg_rating) * item_features[pos, 2:]
            count += 1
        except KeyError:
            missing += 1
    # take the average
    user_vector /= count
    # manually overwrite the first feature to be 1.0 (the max value) as the desired IMDb rating (TODO)
    user_vector[0] = 1.0
    # clip vector to maximum 1 and minimum -1 to optimize cosine similarity (TODO)
    user_vector[2:] = np.clip(user_vector[2:], -1.0, 1.0)
    if missing > 0:
        print(f'Warning: {missing} movies out of {len(user_ratings)} were missing.')
    return user_vector


def calculate_similarity(user_features: np.array, item_features: np.array, top_K=None, threshold=None):
    """ Calculates cosine similarity or cosine distance between the user's feature vector and
        ALL item feature vectors, then orders items based on it. We suggest the most similar movies.
        LSH is typically used to speed this up. """

    # calculate cosine similarity or distance between the user's vector/matrix and all the movies' vectors.
    cos_sim = user_features @ item_features.T         # takes dot product between each item vector and the user vector
    cos_sim /= np.linalg.norm(user_features)          # normalize by the magnitude of user vector
    cos_sim /= np.linalg.norm(item_features, axis=1)  # normalize by the magnitude of item vectors respectively
    print(cos_sim)

    # order by similarity/distance and keep topK most similar or those above/below a threshold
    ordered_pos = (-cos_sim).argsort()
    if top_K is not None:
        ordered_pos = ordered_pos[:top_K]
    elif threshold is not None:
        keep = np.count_nonzero(cos_sim >= threshold)
        ordered_pos = ordered_pos[:keep]

    # TODO (EXTRA): Can we speed this up with black-box LSH or something?

    return cos_sim, ordered_pos


def evaluate(item_features: np.array, movie_pos: dict):
    # load movieLens user ratings
    print('Loading movieLens user ratings...')
    user_ratings: pd.DataFrame = load_user_ratings(movielens_data_folder, limit=1000)
    print(user_ratings)
    print('Done')

    # create user features one-by-one (TODO: slow but easier)
    num_users = user_ratings['userId'].max()
    user_features = np.zeros((num_users, item_features.shape[1]))
    for i in tqdm(range(num_users), total=num_users):
        user_rating = create_user_rating_dict(user_ratings[user_ratings['userId'] == i + 1])
        user_features[i, :] = build_user_feature_vector(user_rating, movie_pos, item_features)
    print(user_features)


def recommend_for_single_user(user_rating: dict, item_features: np.array, movie_pos: bidict, ignore_seen=False, topK=20):
    # build user feature vector
    user_features = build_user_feature_vector(user_rating, movie_pos, item_features)
    print(user_features)

    # make recommendations
    item_features = 2 * item_features - 1   # TODO: Can we use -1 instead of 0? it makes more sense to me but it ruins user feature creation. Maybe add them aposteriori? --> this kinda works I think
    cos_sim, items_pos = calculate_similarity(user_features, item_features, top_K=None if ignore_seen else topK)
    print('min:', min(cos_sim), 'max:', max(cos_sim))
    k = 1
    for pos in items_pos:
        movie_id = movie_pos.inverse[pos]
        if ignore_seen and movie_id in user_rating:
            continue
        print(f'{k}.', movie_id, 'with similarity', cos_sim[pos], '(seen)' if movie_id in user_rating else '')
        k += 1
        if k > topK: break


if __name__ == '__main__':
    load_item_features = True

    if not load_item_features:
        # load rdf
        print('Loading rdf...')
        rdf = load_rdf()
        print('done')

        # extract item features
        movie_pos, item_features = build_items_feature_vetors(rdf)
        print(item_features)
    else:
        # load saved features
        item_features = np.load('item_features.npy')
        movie_pos: bidict = load_dict('movie_pos')

        print(item_features)

    # Manually test a user rating input (STAR WARS)
    user_rating = {
        'tt0076759': 5.0,
        'tt0080684': 4.5,
        'tt0120915': 3.5,
        'tt0121765': 4.5,
        'tt0121766': 4.5
    }

    recommend_for_single_user(user_rating, item_features, movie_pos)

    # evaluate(item_features, movie_pos)
