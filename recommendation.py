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

load_item_features = False        # load previously constructed features or extract them from saved rdf graph
tune = False                      # test different configurations of hyperparameters and keep the best one


# data augmentations
use_wikidata = True               # use extra features from wikidata or not ->  recommended
use_nlp_emotions = True           # use extra features from NLP enrichment  ->  not recommended


""" HYPERPARAMETERS """
override_rating_to_best = True
use_median_for_year = True
use_user_mean_rating = False     # Note: this gives slightly better results for the dataset but doesn't make as much sense for our application
temperature = 50                 # boost for categorical features to be more towards their clipping values (e.g. -1 and 1) so e.g. less good reviews are need to reach the maximum value for an actor
clip = True

# weights (higher weight -> more important), we don't want numerical features to get overshadowed by categorical ones
# Note: if the weights change we have to rebuild the graph! So set load_item_features = False.
w_rating = 1
w_date = 0.25
w_genres = 1
w_actors = 3
w_directors = 2
w_series = 5
w_subjects = 2
w_distributors = 0.1
w_emotions = 0.5


def extract_binary_features(actual_values: set, ordered_possible_values: list) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform([actual_values])
    return binary_format


# noinspection PyUnboundLocalVariable
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
    LEAST_MOVIES2 = 5
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

    if use_wikidata:
        LEAST_MOVIES_SERIES = 2
        LEAST_MOVIES3 = 3

        print('Looking up series...')
        all_series = rdf.query(
            """ SELECT DISTINCT ?series
                WHERE {
                    ?movie pred:hasSeries ?series . 
                }
                GROUP BY ?series
                HAVING (COUNT(?movie) >= """ + str(LEAST_MOVIES_SERIES) + ')',
            initNs={'pred': ns_predicates})
        all_series = sorted([str(s['series']) for s in all_series])
        print('Found', len(all_series), 'series with at least', LEAST_MOVIES_SERIES, 'movies.')

        print('Looking up subjects...')
        all_subjects = rdf.query(
            """ SELECT DISTINCT ?subject
                WHERE {
                    ?movie pred:hasSubject ?subject . 
                }
                GROUP BY ?subject
                HAVING (COUNT(?movie) >= """ + str(LEAST_MOVIES3) + ')',
            initNs={'pred': ns_predicates})
        all_subjects = sorted([str(s['subject']) for s in all_subjects])
        print('Found', len(all_subjects), 'subjects with at least', LEAST_MOVIES3, 'movies.')

        print('Looking up distributors...')
        all_dists = rdf.query(
            """ SELECT DISTINCT ?dist
                WHERE {
                    ?movie pred:hasDistributor ?dist . 
                }
                GROUP BY ?dist
                HAVING (COUNT(?movie) >= """ + str(LEAST_MOVIES3) + ')',
            initNs={'pred': ns_predicates})
        all_dists = sorted([str(d['dist']) for d in all_dists])
        print('Found', len(all_dists), 'distributors with at least', LEAST_MOVIES3, 'movies.')

        # Query all movies on rdf and their associated features
        print('Querying movie features...')
        movies = rdf.query(
            """SELECT DISTINCT ?movie ?year ?rating
              (group_concat(distinct ?genre; separator=",") as ?genres)
              (group_concat(distinct ?actor; separator=",") as ?actors)
              (group_concat(distinct ?director; separator=",") as ?directors)
              (group_concat(distinct ?subject; separator=",") as ?subjects)
              (group_concat(distinct ?serie; separator=",") as ?series)
              (group_concat(distinct ?dist; separator=",") as ?dists)
            WHERE { 
              ?movie pred:hasYear ?year .
              ?movie pred:hasRating ?rating .
              ?movie pred:hasGenre ?genre .
              ?movie pred:hasDirector ?director .
              ?movie pred:hasActor ?actor .
              OPTIONAL { ?movie pred:hasSubject ?subject_temp . }
              OPTIONAL { ?movie pred:hasSeries ?serie_temp . }
              OPTIONAL { ?movie pred:hasDistributor ?dist_temp . }
              BIND(COALESCE(?subject_temp, "") AS ?subject)
              BIND(COALESCE(?serie_temp, "") AS ?serie)
              BIND(COALESCE(?dist_temp, "") AS ?dist)
            } 
            GROUP BY ?movie ?year ?rating""",
            initNs={'movies': ns_movies,
                    'genres': ns_genres,
                    'pred': ns_predicates,
                    'principals': ns_principals})
        print('Done.')
    else:
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

    feature_lens: dict = {}
    NUM_FEATURES = 2 + len(all_genres) + len(all_actors) + len(all_directors)
    feature_lens['genres'] = len(all_genres)
    feature_lens['actors'] = len(all_actors)
    feature_lens['directors'] = len(all_directors)
    if use_wikidata:
        NUM_FEATURES += len(all_series) + len(all_subjects) + len(all_dists)
        feature_lens['series'] = len(all_series)
        feature_lens['subjects'] = len(all_subjects)
        feature_lens['dists'] = len(all_dists)
    if use_nlp_emotions:
        NUM_FEATURES += 5                      # TODO: hardcoded 5
        feature_lens['emotions'] = 5

    movie_pos: bidict = bidict({})
    print(f'Allocating memory for {len(movies)} x {NUM_FEATURES} array...')
    item_features = np.zeros((len(movies), NUM_FEATURES), dtype=np.float32)   # takes too long :(

    print('Creating item feature matrix...')
    for i, movie_data in tqdm(enumerate(movies), total=len(movies)):
        # add movie_id to parallel vector
        movie_pos[movie_data['movie'].split('/')[-1]] = i   # dict with position in item_features

        # get numerical features
        rating = (float(movie_data['rating']) / 10)
        year = (float(int(movie_data['year']) - min_year) / (max_year - min_year))     # min-max scaling

        # Convert all categorical to binary format
        genres = set(movie_data['genres'].split(','))
        actors = set(movie_data['actors'].split(','))
        directors = set(movie_data['directors'].split(','))
        if use_wikidata:
            series = set(movie_data['series'].split(','))
            subjects = set(movie_data['subjects'].split(','))
            dists = set(movie_data['dists'].split(','))
        if use_nlp_emotions:
            # TODO: get these from the RDF Graph
            emotions = set()

        with warnings.catch_warnings():
            # hide user warnings about ignoredmissing values, ignoring these values is the desired behaviour
            warnings.simplefilter("ignore")
            genres_feat = extract_binary_features(genres, all_genres)
            actors_feat = extract_binary_features(actors, all_actors)
            directors_feat = extract_binary_features(directors, all_directors)
            if use_wikidata:
                series_feat = extract_binary_features(series, all_series)
                subjects_feat = extract_binary_features(subjects, all_subjects)
                dists_feat = extract_binary_features(dists, all_dists)
            if use_nlp_emotions:
                emotions_feat = extract_binary_features(emotions, ['Happy', 'Sad', 'Surprise', 'Fear', 'Angry'])        # TODO: hardcoded

        # Concat all of them into one big feature vector
        item_features[i, 0] = w_rating * rating
        item_features[i, 1] = w_date * year
        item_features[i, 2: 2 + len(all_genres)] = w_genres * genres_feat
        item_features[i, 2 + len(all_genres): 2 + len(all_genres) + len(all_actors)] = w_actors * actors_feat
        item_features[i, 2 + len(all_genres) + len(all_actors): 2 + len(all_genres) + len(all_actors) + len(all_directors)] = w_directors * directors_feat
        last = 2 + len(all_genres) + len(all_actors) + len(all_directors)
        if use_nlp_emotions:
            item_features[i, last: last + 5] = w_emotions * emotions_feat   # TODO: hardcoded 5
            last += 5
        if use_wikidata:
            item_features[i, last: last + len(all_series)] = w_series * series_feat
            item_features[i, last + len(all_series): last + len(all_series) + len(all_subjects)] = w_subjects * subjects_feat
            item_features[i, last + len(all_series) + len(all_subjects):] = w_distributors * dists_feat

    # save the result
    if save:
        print('Saving...')
        np.save(f'item_features{"_wikidata" if use_wikidata else ""}.npy', item_features)
        save_dict(movie_pos, f'movie_pos{"_wikidata" if use_wikidata else ""}')
        save_dict(feature_lens, f'feature_lens{"_wikidata" if use_wikidata else ""}')

    print('Done.')

    return movie_pos, item_features, feature_lens


def build_user_feature_vector(user_ratings: dict, movie_pos: bidict, feature_lens: dict, item_features: np.array, temperature=temperature,
                              override_rating_to_best=override_rating_to_best, use_median_for_year=use_median_for_year, clip=clip,
                              use_user_mean_rating=use_user_mean_rating, verbose=False):
    """ Takes as input a user's ratings on IMDb titles and construct its user vector """
    # Note: higher temperature helps have higher values (e.g. closer to 1 and -1) because typically we might get very low numbers
    if use_user_mean_rating:
        avg_rating = np.mean(list(user_ratings.values()))
        # avg_rating = min(2.5, np.mean(list(user_ratings.values())))
    else:
        avg_rating = 2.5
    user_vector = np.zeros(item_features.shape[1], dtype=np.float64)
    missing = 0
    count = 0
    years = []
    for movie_id, rating in user_ratings.items():
        try:
            pos = movie_pos[movie_id]
            # take the normal average for numerical features
            user_vector[:2] += item_features[pos, :2]
            if use_median_for_year:
                years.append(item_features[pos, 1])
            # use weights based on rating for categorical features
            user_vector[2:] += ((rating - avg_rating) / max(5 - avg_rating, avg_rating)) * item_features[pos, 2:]
            count += 1
        except KeyError:
            missing += 1
    # take the average
    if count > 0:
        user_vector /= count
    user_vector[2:] *= temperature
    # manually overwrite the first feature to be 1.0 (the max value) as the desired IMDb rating
    if override_rating_to_best:
        user_vector[0] = 1.0 * w_rating
    # use median for year in order to avoid situations where the mean is nowhere near the most movies the user watches
    if use_median_for_year:
        user_vector[1] = np.median(years)
    # calculated maximum value of each feature based on weight
    # TODO: Hardcoded 2, if more numerical features this needs to change
    # TODO: this can be calculated outside of this since it is the same for each user
    max_clip = np.ones(item_features.shape[1] - 2)
    weights = [w_genres, w_actors, w_directors]
    lens = [feature_lens['genres'], feature_lens['actors'], feature_lens['directors']]
    if use_nlp_emotions:
        weights.append(w_emotions)
        lens.append(5)
    if use_wikidata:
        weights += [w_series, w_subjects, w_distributors]
        lens += [feature_lens['series'], feature_lens['subjects'], feature_lens['dists']]
    last = 0
    for i in range(len(weights)):
        max_clip[last: last + lens[i]] *= weights[i]
        last += lens[i]
    # clip vector to maximum 1 and minimum -1 to optimize cosine similarity
    if clip:
        min_clip = -max_clip  # symmetric
        user_vector[2:] = np.clip(user_vector[2:], min_clip, max_clip)
    if verbose and missing > 0:
        print(f'Warning: {missing} movies out of {len(user_ratings)} were missing.')
    return user_vector, max_clip


def calculate_similarity(user_features: np.array, item_features: np.array, top_K=None, threshold=None):
    """ Calculates cosine similarity or cosine distance between the user's feature vector and
        ALL item feature vectors, then orders items based on it. We suggest the most similar movies.
        LSH is typically used to speed this up. """

    # calculate cosine similarity or distance between the user's vector/matrix and all the movies' vectors.
    cos_sim = user_features @ item_features.T         # takes dot product between each item vector and the user vector
    cos_sim /= np.linalg.norm(user_features)          # normalize by the magnitude of user vector
    cos_sim /= np.linalg.norm(item_features, axis=1)  # normalize by the magnitude of item vectors respectively

    # order by similarity/distance and keep topK most similar or those above/below a threshold
    ordered_pos = (-cos_sim).argsort()
    if len(cos_sim.shape) < 2:   # can't do this if 2d
        if top_K is not None:
            ordered_pos = ordered_pos[:top_K]
        elif threshold is not None:
            keep = np.count_nonzero(cos_sim >= threshold)
            ordered_pos = ordered_pos[:keep]
    elif len(cos_sim.shape) == 2:
        if top_K is not None:
            ordered_pos = ordered_pos[:, :top_K]
        elif threshold is not None:
            keep = np.count_nonzero(cos_sim >= threshold)
            ordered_pos = ordered_pos[:, :keep]

    return cos_sim, ordered_pos


def modify_item_features(item_features: np.array, max_clip):
    new_item_features = np.copy(item_features)
    new_item_features[:, 2:] = 2 * new_item_features[:, 2:] - max_clip      # TODO: Hardcoded 2, if more numerical features this needs to change
    return new_item_features

def evaluate(item_features: np.array, movie_pos: bidict, feature_lens: dict, rating_threshold=3.5, top_K=25,
             limit=100000, use_only_known_ratings=True, print_stats=True, temp=temperature):
    """ Depending on use_only_known_ratings we consider all items or only the items the user has rated and hence knows about.
        Limit has to be set to fit all tables in memory.
    """

    # load movieLens user ratings
    print('Loading movieLens user ratings...')
    user_ratings: pd.DataFrame = load_user_ratings(movielens_data_folder, limit=limit)
    # print(user_ratings)
    print('Done')

    # create user features one-by-one (perhaps slower than a vectorized, but easier
    num_users = user_ratings['userId'].max() - 1   # -1 -> ignore last user which may have incomplete data
    print(f'Found {num_users} users')
    user_features = np.zeros((num_users, item_features.shape[1]))
    user_ratings_dicts = []
    relevant_movies_per_user = []
    max_clip = None
    for i in tqdm(range(num_users), total=num_users, desc='Creating user features'):
        user_rating = create_user_rating_dict(user_ratings[user_ratings['userId'] == i + 1])
        user_features[i, :], max_clip = build_user_feature_vector(user_rating, movie_pos, feature_lens, item_features, temperature=temp)    # Note: max_clip is always the same, we just need one
        user_ratings_dicts.append(user_rating)
        relevant_movies_per_user.append({m for m, r in user_rating.items() if r >= rating_threshold})
    assert(max_clip is not None)
    # print(user_features)

    if print_stats:
        nonzero = np.count_nonzero(user_features, axis=1)
        print(nonzero.shape)
        mean = np.mean(nonzero)
        print('Average non_zero features in user_features are:', f'{mean:.2f}', 'out of', item_features.shape[1], f'({100 * mean / item_features.shape[1]:.2f}%)')

    print('Calculating similarity...')
    item_features = modify_item_features(item_features, max_clip)
    cos_sim, ordered_pos = calculate_similarity(user_features, item_features, top_K=top_K if not use_only_known_ratings else None)
    # print(cos_sim)

    # Formula from: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
    avg_recalls = []
    avg_precisions = []
    for i in tqdm(range(num_users), total=num_users, desc='Calculating MAP and recall'):
        m = len(relevant_movies_per_user[i])
        if m == 0:  # if there are no relevant movies for a user then don't include him
            continue
        avg_precision = 0.0
        num_relevant = 0
        no_knowledge = 0
        N = top_K if not use_only_known_ratings else item_features.shape[0]
        for k in range(N):
            # if k-th item is relevant
            k_movie_id = movie_pos.inverse[ordered_pos[i, k]]
            if k_movie_id in relevant_movies_per_user[i]:
                num_relevant += 1
                precision = num_relevant / (k - no_knowledge + 1)
                avg_precision += precision
            elif use_only_known_ratings and k_movie_id not in user_ratings_dicts[i]:
                # if not a movie on which we have a rating then don't consider it as a part of our top_K
                no_knowledge += 1
            if k - no_knowledge + 1 >= top_K: break      # if found top_K rated movies then stop
        if top_K - no_knowledge == 0:  # can't tell
            continue
        # precision = # of our recommendations that are relevant / # of items we recommended
        avg_precision /= min(top_K, m)
        avg_precisions.append(avg_precision)
        # recall = # of our recommendations that are relevant / # of all the possible relevant items
        recall = num_relevant / min(top_K, m)
        avg_recalls.append(recall)
    print(avg_precisions)
    print(avg_recalls)
    print(f'MAP @ {top_K} = {np.mean(avg_precisions)}')
    print(f'Average recall = {np.mean(avg_recalls)}')
    print(f'{len(avg_precisions)} out of {num_users} users were used')


def recommend_for_single_user(user_rating: dict, item_features: np.array, movie_pos: bidict, feature_lens: dict, ignore_seen=False, topK=20, verbose=True):
    # build user feature vector
    user_features, max_clip = build_user_feature_vector(user_rating, movie_pos, feature_lens, item_features, verbose=True)
    print('user_features =', user_features)

    # make recommendations
    item_features = modify_item_features(item_features, max_clip)
    cos_sim, items_pos = calculate_similarity(user_features, item_features, top_K=None if ignore_seen else topK)
    print('min:', min(cos_sim), 'max:', max(cos_sim))
    k = 1

    recommendations = []
    for pos in items_pos:
        movie_id = movie_pos.inverse[pos]
        if ignore_seen and movie_id in user_rating:
            continue
        recommendations.append({'tconst': movie_id, 'similarity': cos_sim[pos]})
        if verbose:
            print(f'{k}.', movie_id, 'with similarity', cos_sim[pos], '(seen)' if movie_id in user_rating else '')
        k += 1
        if k > topK: break

    return recommendations


def tune_hyperparameters():
    global w_rating, w_date, w_genres, w_actors, w_directors, w_series, w_subjects, w_distributors, temperature

    # load rdf
    print('Loading rdf...')
    rdf = load_rdf()
    print('done')

    confs = [
        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": 1},
        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": .2},
        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": .1},

        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": .1},
        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 2, "w_subjects": 1, "w_distributors": .1},
        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 3, "w_subjects": 1, "w_distributors": .1},
        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 5, "w_subjects": 1, "w_distributors": .1},

        # {"w_rating": 0.5, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": .1},
        # {"w_rating": 1, "w_date": 0.5, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": .1},
        # {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": .1},
        # {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": .1},

        # {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 2, "w_directors": 2, "w_series": 1, "w_subjects": 1, "w_distributors": 0.1},
        # {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 3, "w_directors": 2, "w_series": 1.5, "w_subjects": 1, "w_distributors": 0.1},
        # {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 3, "w_directors": 2, "w_series": 1, "w_subjects": 1, "w_distributors": 0.1},

        {"w_rating": 1, "w_date": 1, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": 1},
        {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 1, "w_directors": 1, "w_series": 1, "w_subjects": 1, "w_distributors": 0.1},
        {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 3, "w_directors": 2, "w_series": 1, "w_subjects": 1, "w_distributors": 0.1},
        {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 3, "w_directors": 2, "w_series": 1, "w_subjects": 2, "w_distributors": 0.1},
        {"w_rating": 1, "w_date": 0.25, "w_genres": 1, "w_actors": 3, "w_directors": 2, "w_series": 1.5, "w_subjects": 2, "w_distributors": 0.1},
        {"w_rating": 0.5, "w_date": 0.25, "w_genres": 1.5, "w_actors": 3, "w_directors": 2, "w_series": 1.5, "w_subjects": 2, "w_distributors": 0.1},
    ]

    for conf in confs:
        w_rating, w_date, w_genres, w_actors, w_directors, w_series, w_subjects, w_distributors = conf.values()

        # extract item features
        movie_pos, item_features, feature_lens = build_items_feature_vetors(rdf, save=False)
        print(f'Built {item_features.shape[0]} movies with {item_features.shape[1]} features each.')

        # evaluate for this conf for a bunch of temperatures
        for temperature in [50]:  # [1, 10, 25, 50, 100]:
            print(f'> Temperature: {temperature} Conf:', w_rating, w_date, w_genres, w_actors, w_directors, w_series, w_subjects, w_distributors)
            evaluate(item_features, movie_pos, feature_lens, temp=temperature)
            print('')


if __name__ == '__main__':
    if tune:
        tune_hyperparameters()
    else:
        if not load_item_features:
            # load rdf
            print('Loading rdf...')
            rdf = load_rdf()
            print('done')

            # extract item features
            movie_pos, item_features, feature_lens = build_items_feature_vetors(rdf)
            print(f'Built {item_features.shape[0]} movies with {item_features.shape[1]} features each.')
        else:
            # load saved features
            print('Loading previously saved features...')
            item_features = np.load(f'item_features{"_wikidata" if use_wikidata else ""}.npy')
            movie_pos: bidict = load_dict(f'movie_pos{"_wikidata" if use_wikidata else ""}')
            feature_lens: dict = load_dict(f'feature_lens{"_wikidata" if use_wikidata else ""}')

            print(f'Loaded {item_features.shape[0]} movies with {item_features.shape[1]} features each.')

        # Manually test a user rating input (STAR WARS)
        user_rating = {
            'tt0076759': 5.0,
            'tt0080684': 4.5,
            'tt0120915': 3.5,
            'tt0121765': 4.5,
            'tt0121766': 4.5
        }

        recommend_for_single_user(user_rating, item_features, movie_pos, feature_lens)

        evaluate(item_features, movie_pos, feature_lens)
