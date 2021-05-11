import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from rdflib import Graph
from tqdm import tqdm

from rdf import ns_movies, ns_genres, ns_predicates, ns_principals, load_rdf


def extract_binary_features(actual_values: set, ordered_possible_values: list) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform([actual_values])
    return binary_format


def build_items_feature_vetors(rdf: Graph) -> (list, np.array):   # list parallel to 2d np array's columns --> the title ID for the row
    """ Builds a feature vector for each movie """
    # get all possible categorical features
    all_genres = rdf.query(
        """ SELECT DISTINCT ?genre
            WHERE {
                ?movie pred:hasGenre ?genre . 
            }""", initNs={'pred': ns_predicates})
    all_genres = sorted([str(g['genre']) for g in all_genres])
    # print(all_genres)

    # TODO: Ignore insignificant actors somehow?
    all_actors = rdf.query(
        """ SELECT DISTINCT ?actor
            WHERE {
                ?movie pred:hasActor ?actor . 
            }""", initNs={'pred': ns_predicates})
    all_actors = sorted([str(a['actor']) for a in all_actors])
    # Note: keep just the id with: actors = sorted([str(a['actor']).split('/')[-1] for a in actors])
    # print(all_actors)

    all_directors = rdf.query(
        """ SELECT DISTINCT ?director
            WHERE {
                ?movie pred:hasDirector ?director . 
            }""", initNs={'pred': ns_predicates})
    all_directors = sorted([str(d['director']) for d in all_directors])
    # print(all_directors)

    # Query all movies on rdf and their associated features
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
           GROUP BY ?movie ?year ?rating
           LIMIT 10""",   # TODO: remove LIMIT
        initNs={'movies': ns_movies,
                'genres': ns_genres,
                'pred': ns_predicates,
                'principals': ns_principals})

    NUM_FEATURES = 2 + len(all_genres) + len(all_actors) + len(all_directors)  # TODO
    movie_ids: [str] = [''] * len(movies)
    item_features = np.zeros((len(movies), NUM_FEATURES), dtype=np.float32)

    for i, movie_data in tqdm(enumerate(movies), total=len(movies)):
        # add movie_id to parallel vector
        movie_ids[i] = movie_data['movie']

        # get numerical features
        rating = float(movie_data['rating'])
        year = float(movie_data['year'])     # TODO: add a factor?

        # Convert all categorical to binary format
        genres = set(movie_data['genres'].split(','))
        actors = set(movie_data['actors'].split(','))
        directors = set(movie_data['directors'].split(','))
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
        print(item_features[i])

    # TODO: save the result?

    return movie_ids, item_features


def build_user_feature_vector(rdf: Graph, user_ratings: dict, item_features: np.array or pd.DataFrame):
    """ Takes as input a user's ratings on IMDb titles and construct its user vector """
    # TODO: Query all the movies that the user has rated
    # TODO: Get their feature vectors as saved from earlier
    # TODO: Aggregate all these feature vectors to create the user's feature vector
    #       Find the average rating the user gives and subtract it in order to have positive and negative normalized features (~ see MMD again for this process)
    pass


def recommend_movies(user_vector, item_features: np.array or pd.DataFrame, top_K=None, threshold=None):
    """ Calculates cosine similarity or cosine distance between the user's feature vector and
        ALL item feature vectors, then orders items based on it. Suggest the most similar movies.
        LSH is typically used to speed this up. """
    # TODO: calculate cosine similarity or distance between the user's vector and all the movies' vectors
    # TODO: order by similarity/distance
    # TODO: return topK most similar or those above/below a threshold
    # TODO (EXTRA): Can we speed this up with black-box LSH or something?
    pass


if __name__ == '__main__':
    # load rdf
    print('Loading rdf...')
    rdf = load_rdf()
    print('done')

    movie_ids, item_features = build_items_feature_vetors(rdf)
    print(movie_ids, item_features)
