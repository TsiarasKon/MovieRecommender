import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from rdflib import Graph


def extract_binary_features(actual_values: [set], ordered_possible_values: list) -> np.array:
    """ Converts a categorical feature with multiple values to a multi-label binary encoding """
    mlb = MultiLabelBinarizer(classes=ordered_possible_values)
    binary_format = mlb.fit_transform(actual_values)
    return binary_format


def build_items_feature_vetors(rdf: Graph) -> (list, np.array):   # list parallel to 2d np array's columns --> the title ID for the row
    """ Builds a feature vector for each movie """
    # TODO: Query all movies on rdf
    # TODO: Get their associated features
    # TODO: Convert all categorical to binary format
    # TODO (EXTRA): Add a different weight to each feature with which to experiment "balancing"?  How do we change this afterwards? Must also store it?
    # TODO: Concat all of them into one big feature vector
    # TODO: return and/or save the result
    pass


def build_user_feature_vector(rdf: Graph, user_ratings: dict, item_features: np.array or pd.DataFrame):
    """ Takes as input a user's ratings on IMDb titles and construct its user vector """
    # TODO: Query all the movies that the user has rated
    # TODO: Get their feature vectors as saved from earlier
    # TODO: Aggregate all these feature vectors to create the user's feature vector
    #       Find the average rating the user gives and subtract it in order to have positive and negative features (~ see MMD again for this process)
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
    # example
    res = extract_binary_features([{'action', 'comedy'}], ['comedy', 'drama', 'action'])
    print(res)
