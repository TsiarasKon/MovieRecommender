import pickle
import pandas as pd
import numpy as np


""" save and load python dictionary """
def save_dict(dict, name):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)


def load_user_ratings(movielens_data_folder, limit=None) -> pd.DataFrame:
    # load movielens user reviews data
    user_ratings = pd.read_csv(movielens_data_folder + 'ratings.csv',
                               usecols=['userId', 'movieId', 'rating'],
                               dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float32})
    if limit is not None:
        user_ratings = user_ratings[:limit]
    # link movieIds with imdbIds
    links = pd.read_csv(movielens_data_folder + 'links.csv',
                        index_col='movieId',
                        usecols=['movieId', 'imdbId'],
                        dtype={'movieId': np.int32, 'imdbId': 'string'})
    user_ratings['movieId'] = 'tt' + user_ratings['movieId'].map(links['imdbId'])
    return user_ratings


def create_user_rating_dict(user_ratings: pd.DataFrame):
    r = {}
    for _, user_rating in user_ratings.iterrows():
        r[user_rating['movieId']] = user_rating['rating']
    return r
